"""Data-space iteration rollout (arXiv:2610.12102, Eq. 8) and the DMD noise sampler.

Data-space rollout: the source noise z is held fixed across all steps, flow
time is pinned to t = 1000 (sigma' = 1), and the state iterates in pixel space:

    x0_hat^(0) = 0;   x0_hat^(i+1) = z - v_gen(z, x0_hat^(i), r=i)

Every intermediate state is a clean, unnoised data estimate.
"""

import torch
from iris3b.flow.schedule import shift_sigma

from ..models.lora import use_adapter


def rollout_dataspace(model, z, y, y_mask, r: int, track_grad: bool = True) -> torch.Tensor:
    """Run r+1 generator steps from zero; grad flows only through the last.

    Caller must run ``backward()`` inside ``use_adapter(model, "gen")`` for
    activation-checkpointing correctness.
    """
    x_hat = torch.zeros_like(z)
    t_full = torch.full((z.shape[0],), 1000.0, dtype=torch.float32, device=z.device)
    with use_adapter(model, "gen"):
        for i in range(r + 1):
            with torch.set_grad_enabled(track_grad and i == r):
                v = model(z, t_full, y, x0=x_hat, r=i, y_mask=y_mask).x
                x_hat = z - v
    return x_hat


def sample_shifted_sigma(batch: int, shift: float, device, clamp=(0.02, 0.98)) -> torch.Tensor:
    """DMD noising time: tau ~ Beta(2.5, 1) mapped through Iris's shift.

    Beta(2.5, 1) concentrates on high noise (mean ~0.71), matching the paper's
    choice; the shift maps to the noise level the network is conditioned on
    (t = 1000 * sigma'). Clamped after the shift (PLAN.md's convention).
    """
    u = torch.distributions.Beta(2.5, 1.0).sample((batch,))
    return shift_sigma(u, shift).clamp(*clamp).to(device)