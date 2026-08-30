"""
Forward-backward consistency -> covisibility for plain optical-flow networks.

Given the forward flow ``f`` (src -> tgt) and the backward flow ``b`` (tgt -> src)
the cycle error at a source pixel ``x`` is ``c(x) = f(x) + b(x + f(x))``.  The
classic occlusion criterion of Sundaram, Brox & Keutzer (ECCV 2010) marks ``x``
as *occluded* when

    |c|^2 >= alpha * (|f|^2 + |b(x+f)|^2) + beta,      alpha = 0.01, beta = 0.5

We expose the continuous ratio ``r = |c|^2 / (alpha (|f|^2 + |b|^2) + beta)`` as a
soft covisibility score ``cov = clip(1 - r / 2, 0, 1)``; therefore
``cov > 0.5  <=>  r < 1  <=>  the classic "visible" decision``, which makes the
default ``--covis_thresh 0.5`` of the metric meaningful for every backend.  Pixels
whose forward match falls outside the target image get ``cov = 0``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def coords_grid(h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(1, 2, h, w) pixel coordinate grid (x, y)."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij"
    )
    return torch.stack([xs, ys], dim=0)[None]


def sample_at_flow(field_b_c_ht_wt: torch.Tensor, flow_fwd_b2hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample a target-grid field at ``x + f(x)`` for every source pixel.

    Returns ``(sampled (B,C,H,W), in_bounds (B,H,W) bool)``.  Uses
    ``align_corners=True`` pixel-centre semantics (pixel ``i`` <-> coordinate ``i``).
    """
    B, _, H, W = flow_fwd_b2hw.shape
    Ht, Wt = int(field_b_c_ht_wt.shape[-2]), int(field_b_c_ht_wt.shape[-1])
    grid = coords_grid(H, W, flow_fwd_b2hw.device, flow_fwd_b2hw.dtype).expand(B, -1, -1, -1) + flow_fwd_b2hw
    u, v = grid[:, 0], grid[:, 1]
    in_bounds = (u >= 0) & (u <= Wt - 1) & (v >= 0) & (v <= Ht - 1)
    u_n = 2.0 * u / max(Wt - 1, 1) - 1.0
    v_n = 2.0 * v / max(Ht - 1, 1) - 1.0
    norm_grid = torch.stack([u_n, v_n], dim=-1)  # (B,H,W,2)
    sampled = F.grid_sample(field_b_c_ht_wt, norm_grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled, in_bounds


def forward_backward_covisibility(
    flow_fwd_b2hw: torch.Tensor,
    flow_bwd_b2hw: torch.Tensor,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft covisibility from forward/backward flows.

    Args:
        flow_fwd_b2hw: (B, 2, H, W)  src -> tgt, pixels of the source grid.
        flow_bwd_b2hw: (B, 2, Ht, Wt) tgt -> src, pixels of the target grid.

    Returns:
        ``(cov (B,H,W) in [0,1], cycle_error (B,H,W) pixels, in_bounds (B,H,W) bool)``.
    """
    bwd_at_fwd, in_bounds = sample_at_flow(flow_bwd_b2hw, flow_fwd_b2hw)
    cycle = flow_fwd_b2hw + bwd_at_fwd
    err2 = (cycle**2).sum(1)
    mag2 = (flow_fwd_b2hw**2).sum(1) + (bwd_at_fwd**2).sum(1)
    ratio = err2 / (alpha * mag2 + beta)
    cov = (1.0 - 0.5 * ratio).clamp(0.0, 1.0)
    cov = torch.where(in_bounds, cov, torch.zeros_like(cov))
    return cov, err2.sqrt(), in_bounds


__all__ = ["coords_grid", "forward_backward_covisibility", "sample_at_flow"]
