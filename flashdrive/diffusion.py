"""Adaptive action caching for FlashDrive's diffusion sampling.

``euler_with_cache`` reuses the predicted velocity on the configured
``cache_steps`` to skip action-expert forwards; ``FlashDriveBaseMixin._action``
dispatches to it when ``diffusion_kwargs`` requests ``int_method="euler_with_cache"``.
"""

from collections.abc import Callable
from typing import Any

import torch


def euler_with_cache(
    diffusion: Any,
    *,
    noise: torch.Tensor,
    batch_size: int,
    step_fn: Callable[..., torch.Tensor],
    cache_steps: list[int],
    device: torch.device,
    inference_step: int,
) -> torch.Tensor:
    """Euler integration that reuses the cached velocity on ``cache_steps``.

    ``diffusion`` is the model's ``FlowMatching`` instance; it provides the
    action shape (``x_dims``) and carries the persistent velocity buffer.
    """
    if 0 in cache_steps:
        raise ValueError("cache_steps cannot include step 0: no velocity is cached yet.")
    x = noise
    time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device, dtype=torch.bfloat16)
    n_dim = len(diffusion.x_dims)

    for i in range(inference_step):
        dt = time_steps[i + 1] - time_steps[i]  # scalar; broadcasts in the update
        t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
        if i in cache_steps:
            v = diffusion._cached_velocity
        else:
            v = step_fn(x=x, t=t_start)
            if not hasattr(diffusion, "_cached_velocity"):
                diffusion._cached_velocity = torch.empty_like(v)
            diffusion._cached_velocity.copy_(v)
        x = x + dt * v
    return x
