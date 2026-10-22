"""Data-space DMD losses (arXiv:2610.12102, Eq. 10-11).

Both loss functions return the scalar; the CALLER runs ``backward()`` inside
the matching ``use_adapter`` context ("gen" for the generator loss, "fake" for
the critic) so activation-checkpoint recompute sees the same live adapter.
"""

import torch
import torch.nn.functional as F

from ..models.lora import use_adapter
from .rollout import rollout_dataspace

MODEL_TIME_SCALE = 1000.0  # model time = 1000 * shifted sigma'


def _per_sample(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return sigma.view(-1, *([1] * (like.ndim - 1)))


def teacher_velocity_windowed(
    model, x_tau, sigma, y, y_mask, null, null_mask, cfg_scale=3.0, cfg_window=0.6
) -> torch.Tensor:
    """Teacher velocity with the paper's guidance window (tau in [0.0, 0.6]).

    CFG where the shifted sigma' <= cfg_window; plain conditional velocity
    above it (prevents color blooming at high noise). Cond and uncond passes
    run sequentially to halve peak memory vs a doubled batch.
    """
    t_model = sigma * MODEL_TIME_SCALE
    with torch.no_grad():
        with use_adapter(model, "off"):
            v_cond = model(x_tau, t_model, y, y_mask=y_mask).x.float()
            v_uncond = model(x_tau, t_model, null, y_mask=null_mask).x.float()
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
    return torch.where(_per_sample(sigma, x_tau) <= cfg_window, v_cfg, v_cond)


def dmd_generator_loss(
    model, x0, sigma, noise, y, y_mask, null, null_mask, r, cfg_scale=3.0, cfg_window=0.6
) -> torch.Tensor:
    """Eq. 10 with w(tau)=1: sg[v_pos - v_fake] . x0, summed over pixels.

    ``x0`` must carry the generator graph (the rollout's last-step output).
    Sum over pixel dims (not mean) keeps gradients away from Adam's eps floor.
    The MSE-style mean-reduction of the critic loss makes loss SCALE
    inconsequential for Adam, so w(tau)=1 plain is used here too.
    """
    per = _per_sample(sigma, x0)
    x_tau = (1 - per) * x0 + per * noise  # Eq. 9; graph kept through x0
    v_pos = teacher_velocity_windowed(model, x_tau, sigma, y, y_mask, null, null_mask, cfg_scale, cfg_window)
    with torch.no_grad():
        with use_adapter(model, "fake"):
            v_fake = model(x_tau, sigma * MODEL_TIME_SCALE, y, r=r, y_mask=y_mask).x.float()
    direction = (v_pos - v_fake).detach()
    return (x0 * direction).flatten(1).sum(1).mean()


def generator_loss(
    model, z, y, y_mask, r, sigma, noise, null, null_mask, cfg_scale=3.0, cfg_window=0.6
) -> torch.Tensor:
    """Rollout (grad on last step) + DMD loss."""
    x0 = rollout_dataspace(model, z, y, y_mask, r, track_grad=True)
    return dmd_generator_loss(model, x0, sigma, noise, y, y_mask, null, null_mask, r, cfg_scale, cfg_window)


def fake_critic_loss(model, z, y, y_mask, r, sigma, noise) -> torch.Tensor:
    """Eq. 11: flow loss of the fake net on detached generator rollouts.

    Target velocity is eps - x0_hat with the SAME eps that built x_tau.
    Mean-reduction (standard flow-matching convention, same as turbo's critic).
    """
    with torch.no_grad():
        x0 = rollout_dataspace(model, z, y, y_mask, r, track_grad=False)
    per = _per_sample(sigma, x0)
    x_tau = (1 - per) * x0 + per * noise
    with use_adapter(model, "fake"):
        v_fake = model(x_tau, sigma * MODEL_TIME_SCALE, y, r=r, y_mask=y_mask).x
    return F.mse_loss(v_fake.float(), (noise - x0).float())