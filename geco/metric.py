"""
Backend-agnostic GeCo error computation.

``demo_detection.py`` and ``GeCo-Eval_evaluation.py`` share exactly the same
per-pair logic; it lives here so that swapping the geometry / flow backend
never touches the metric itself.  For one (source, target) pair:

1. *Ego-motion flow*: project the source depth into the target camera
   (``rigid_flow_from_camera_motion``) -> ``ego`` and its validity ``mask_rp``.
2. *Motion error* ``e_xy``: ``|flow - ego|`` normalised by the focal length
   (unit-less, ~ radians).  Valid where the flow model says "covisible" and the
   re-projection is valid.
3. *Structure error* ``e_z``: relative depth inconsistency between the
   projected source depth and the target depth sampled at the projected
   location (bidirectional check for the demo, unidirectional for the benchmark).
4. *Fusion*: average both errors on non-occluded pixels, use ``e_z`` alone on
   "wrong occlusion" pixels (flow says occluded, geometry says visible), and
   ``e_xy`` alone where only the motion term is valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from geco.flow.base import FlowOutput
from geco.geometry.base import GeometryOutput
from utils import (
    compute_normalized_depth_error_bidirectional,
    compute_normalized_depth_error_unidirectional,
    masked_mean,
    normalize_flow_to_unitless,
    rigid_flow_from_camera_motion,
)

DEPTH_CHECKS = ("bidirectional", "unidirectional")


def source_confidence_mask(
    geo: GeometryOutput,
    idx: int,
    conf_percentile: float,
    conf_min: float,
) -> torch.Tensor:
    """(H, W) bool mask of confidently-predicted source pixels.

    A pixel is kept when its confidence is >= the ``conf_percentile``-th
    percentile of the frame's *valid* confidences **and** > ``conf_min``.
    (Identical to ``create_confidence_mask_torch`` for backends that predict
    every pixel; invalid pixels are excluded from the percentile otherwise.)
    """
    conf = geo.conf[idx, ..., 0]
    valid = geo.valid[idx]
    if not bool(valid.any()):
        return torch.zeros_like(valid)
    vals = conf[valid].flatten()
    if vals.numel() > 16_000_000:  # torch.quantile input-size limit
        stride = vals.numel() // 16_000_000 + 1
        vals = vals[::stride]
    thr = torch.quantile(vals, q=float(conf_percentile) / 100.0)
    return valid & (conf >= thr) & (conf > conf_min)


@dataclass
class PairErrors:
    e_xy: torch.Tensor  # (H, W) motion error, 0 where not mot_valid
    e_z: torch.Tensor  # (H, W) structure error, 0 where not dep_mask
    p_fus: torch.Tensor  # (H, W) fused error, 0 where not fus_mask
    mot_valid: torch.Tensor  # (H, W) bool
    dep_mask: torch.Tensor  # (H, W) bool  (non-occluded | wrong-occlusion)
    fus_mask: torch.Tensor  # (H, W) bool
    # diagnostics
    mask_cov: torch.Tensor
    mask_rp: torch.Tensor
    dep_valid: torch.Tensor
    wrong_occ: torch.Tensor
    ego_flow: torch.Tensor  # (H, W, 2)


def compute_pair_errors(
    geo: GeometryOutput,
    src: int,
    tgt: int,
    flow_out: FlowOutput,
    conf_src_mask: torch.Tensor,
    *,
    covis_thresh: float = 0.5,
    tau_z: float = 0.02,
    depth_check: str = "bidirectional",
) -> PairErrors:
    if depth_check not in DEPTH_CHECKS:
        raise ValueError(f"depth_check must be one of {DEPTH_CHECKS}, got {depth_check!r}")
    K, E, D = geo.intrinsic, geo.extrinsic, geo.depth
    H, W = geo.image_hw
    if tuple(flow_out.flow.shape[:2]) != (H, W):
        raise ValueError(f"flow grid {tuple(flow_out.flow.shape[:2])} != geometry grid {(H, W)}")

    # 1. ego-motion flow from depth + relative pose
    ego, mask_rp = rigid_flow_from_camera_motion(D[src], K[[src, tgt]], E[[src, tgt]])
    mask_rp = mask_rp & conf_src_mask

    # 2. motion residual (flow model vs. camera-induced flow)
    cov = flow_out.covisibility
    mask_cov = flow_out.valid & torch.isfinite(cov) & (cov > covis_thresh)
    res_uv = normalize_flow_to_unitless(flow_out.flow - ego, K[src])
    e_xy = torch.linalg.vector_norm(res_uv, dim=-1)

    # 3. structure (depth re-projection) error
    fn = compute_normalized_depth_error_bidirectional if depth_check == "bidirectional" else compute_normalized_depth_error_unidirectional
    e_z, val, _, _dz, dz_p = fn(D[src], K[src], E[src], D[tgt], K[tgt], E[tgt], align_corners=True)
    dep_valid = val & conf_src_mask

    # 4. fusion masks
    mot_valid = mask_cov & mask_rp  # flow says visible, geometry re-projects inside the target
    dep_no_occ = mot_valid & dep_valid  # + depth check passes
    wrong_occ = (~mask_cov) & mask_rp & dep_valid & ~(dz_p > tau_z)  # flow says occluded, geometry disagrees
    dep_mask = dep_no_occ | wrong_occ
    mot_only = mot_valid & ~dep_no_occ
    fus_mask = dep_mask | mot_only

    zero = torch.zeros((), device=e_xy.device, dtype=e_xy.dtype)
    e_xy = torch.where(mot_valid, e_xy, zero)
    e_z = torch.where(dep_mask, e_z, zero)
    p_fus = torch.zeros_like(e_xy)
    p_fus = torch.where(dep_no_occ, 0.5 * (e_xy + e_z), p_fus)
    p_fus = torch.where(wrong_occ, e_z, p_fus)
    p_fus = torch.where(mot_only, e_xy, p_fus)

    return PairErrors(
        e_xy=e_xy, e_z=e_z, p_fus=p_fus,
        mot_valid=mot_valid, dep_mask=dep_mask, fus_mask=fus_mask,
        mask_cov=mask_cov, mask_rp=mask_rp, dep_valid=dep_valid, wrong_occ=wrong_occ, ego_flow=ego,
    )


class ErrorAccumulator:
    """Accumulates per-pair errors of one source frame over its neighbours."""

    def __init__(self, hw: Tuple[int, int], device: torch.device | str):
        H, W = hw
        z = lambda: torch.zeros((H, W), device=device, dtype=torch.float32)  # noqa: E731
        self.mot_sum, self.mot_cnt = z(), z()
        self.dep_sum, self.dep_cnt = z(), z()
        self.fus_sum, self.fus_cnt = z(), z()
        self.num_pairs = 0

    def add(self, pe: PairErrors) -> None:
        self.mot_sum += pe.e_xy
        self.mot_cnt += pe.mot_valid.float()
        self.dep_sum += pe.e_z
        self.dep_cnt += pe.dep_mask.float()
        self.fus_sum += pe.p_fus
        self.fus_cnt += pe.fus_mask.float()
        self.num_pairs += 1

    @staticmethod
    def _safe_avg(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return s / c.clamp_min(1.0)

    def maps(self) -> Dict[str, torch.Tensor]:
        """Per-pixel average error maps (0 where no pair contributed) and their counts."""
        return {
            "motion": self._safe_avg(self.mot_sum, self.mot_cnt),
            "depth": self._safe_avg(self.dep_sum, self.dep_cnt),
            "fused": self._safe_avg(self.fus_sum, self.fus_cnt),
            "motion_cnt": self.mot_cnt,
            "depth_cnt": self.dep_cnt,
            "fused_cnt": self.fus_cnt,
        }

    def scalars(self) -> Tuple[float, float, float]:
        """Frame-level (motion, depth, fused) means over pixels that received >= 1 contribution (NaN if none)."""
        m = self.maps()
        return (
            masked_mean(m["motion"], self.mot_cnt > 0),
            masked_mean(m["depth"], self.dep_cnt > 0),
            masked_mean(m["fused"], self.fus_cnt > 0),
        )


__all__ = ["DEPTH_CHECKS", "ErrorAccumulator", "PairErrors", "compute_pair_errors", "source_confidence_mask"]
