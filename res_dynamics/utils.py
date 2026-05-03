from __future__ import annotations

import torch
from torch import Tensor


def compute_tracking_residual_features(
    q_hist: Tensor,
    u_hist: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Build proprioceptive contact features from command-position mismatch.

    Args:
        q_hist: Measured joint position history with shape [B, K, J].
        u_hist: Commanded joint position history with shape [B, K, J].
        eps: Small positive value for stable ratio computation.

    Returns:
        Tensor with shape [B, K, J, 6], ordered as:
        q, u, e = u - q, delta_u, delta_q, rho.
    """
    if q_hist.shape != u_hist.shape:
        raise ValueError(
            f"q_hist and u_hist must have matching shapes, got {q_hist.shape} and {u_hist.shape}"
        )
    if q_hist.ndim != 3:
        raise ValueError(f"q_hist and u_hist must have shape [B, K, J], got {q_hist.shape}")

    e_hist = u_hist - q_hist

    delta_u = torch.zeros_like(u_hist)
    delta_q = torch.zeros_like(q_hist)
    delta_u[:, 1:] = u_hist[:, 1:] - u_hist[:, :-1]
    delta_q[:, 1:] = q_hist[:, 1:] - q_hist[:, :-1]

    rho = delta_q.abs() / (delta_u.abs() + eps)

    # [B, K, J, 6], with scalar per-joint features stacked in the last dimension.
    return torch.stack((q_hist, u_hist, e_hist, delta_u, delta_q, rho), dim=-1)
