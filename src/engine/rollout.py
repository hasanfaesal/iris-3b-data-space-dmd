"""Data-space iteration rollout (arXiv:2610.12102, Eq. 8) and the DMD noise sampler.

Data-space rollout: the source noise z is held fixed across all steps, flow
time is pinned to t = 1000 (sigma' = 1), and the state iterates in pixel space:

    x0_hat^(0) = 0;   x0_hat^(i+1) = z - v_gen(z, x0_hat^(i), r=i)

Every intermediate state is a clean, unnoised data estimate.
"""

import torch
from iris3b.flow.schedule import shift_sigma

from ..models.lora import use_adapter


def rollout_dataspace(
    model, z, y, y_mask, r: int, track_grad: bool = True, return_all_steps: bool = False
) -> torch.Tensor | list[torch.Tensor]:
    """Run r+1 generator steps from zero; grad flows only through the last.

    Caller must run ``backward()`` inside ``use_adapter(model, "gen")`` for
    activation-checkpointing correctness.
    """
    x_hat = torch.zeros_like(z)
    steps = [x_hat] if return_all_steps else None
    t_full = torch.full((z.shape[0],), 1000.0, dtype=torch.float32, device=z.device)
    with use_adapter(model, "gen"):
        for i in range(r + 1):
            with torch.set_grad_enabled(track_grad and i == r):
                v = model(z, t_full, y, x0=x_hat, r=i, y_mask=y_mask).x
                x_hat = z - v
                if return_all_steps:
                    steps.append(x_hat)
    return steps if return_all_steps else x_hat


def sample_shifted_sigma(batch: int, shift: float, device, u_range=(0.02, 0.98)) -> torch.Tensor:
    """DMD noising time: u ~ Uniform(0.02, 0.98) mapped through Iris's shift.

    The shift alone skews toward high noise (density 16x higher at sigma'=1 than
    at 0 for shift 4), landing near the paper's unshifted Beta(2.5, 1) (median
    sigma' 0.80 vs 0.76). Beta on top of the shift skewed twice: 0.3% of rows
    reached sigma' <= 0.3, where within-patch detail is visible.
    """
    u = torch.empty(batch).uniform_(*u_range)
    return shift_sigma(u, shift).to(device)