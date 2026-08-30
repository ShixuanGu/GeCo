"""
Optical-flow backend contract.

A flow backend maps a *source* and a *target* frame (same resolution) to

* ``flow``: (H, W, 2) float32, **pixels** of the source grid, ``(dx, dy)`` such
  that source pixel ``(u, v)`` corresponds to target pixel ``(u + dx, v + dy)``;
* ``covisibility``: (H, W) float32 in ``[0, 1]``; 1 = the source pixel is visible
  (non-occluded and inside the target image), 0 = occluded / out of view.  The
  metric thresholds it with ``--covis_thresh`` (default 0.5).
  Models with a native covisibility head (UFM) return it directly; for plain
  flow networks (RAFT family, PTLFlow zoo) it is derived from a forward-backward
  consistency check, see :mod:`geco.flow.consistency` -- the mapping is chosen so
  that ``covisibility > 0.5`` is exactly the classic Sundaram et al. criterion
  ``|f + b(x+f)|^2 < 0.01 (|f|^2 + |b(x+f)|^2) + 0.5``.
* ``valid``: (H, W) bool, finite flow and covisibility.

Inputs may be ``uint8`` RGB numpy arrays (H, W, 3) -- the common case when frames
are read from disk -- or float tensors in ``[0, 1]`` (H, W, 3) / (3, H, W), which
keeps the computation differentiable for backends with ``supports_grad = True``.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ImageLike = Union[np.ndarray, torch.Tensor]


@dataclass
class FlowOutput:
    flow: torch.Tensor  # (H, W, 2) pixels, src -> tgt
    covisibility: torch.Tensor  # (H, W) in [0, 1]
    valid: torch.Tensor  # (H, W) bool
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def image_hw(self) -> Tuple[int, int]:
        return int(self.flow.shape[0]), int(self.flow.shape[1])

    def validate(self, image_hw: Optional[Tuple[int, int]] = None) -> "FlowOutput":
        if self.flow.ndim != 3 or self.flow.shape[-1] != 2:
            raise ValueError(f"[flow] flow must be (H,W,2), got {tuple(self.flow.shape)}")
        H, W = self.image_hw
        if image_hw is not None and (H, W) != tuple(image_hw):
            raise ValueError(f"[flow] flow grid {(H, W)} != source image grid {tuple(image_hw)}")
        if tuple(self.covisibility.shape) != (H, W):
            raise ValueError(f"[flow] covisibility must be {(H, W)}, got {tuple(self.covisibility.shape)}")
        if tuple(self.valid.shape) != (H, W) or self.valid.dtype != torch.bool:
            raise ValueError(f"[flow] valid must be bool {(H, W)}, got {tuple(self.valid.shape)} {self.valid.dtype}")
        if self.flow.dtype != torch.float32 or self.covisibility.dtype != torch.float32:
            raise ValueError(f"[flow] flow/covisibility must be float32, got {self.flow.dtype}/{self.covisibility.dtype}")
        if len({self.flow.device, self.covisibility.device, self.valid.device}) != 1:
            raise ValueError("[flow] tensors must live on one device")
        with torch.no_grad():
            cov = self.covisibility
            if bool((cov[torch.isfinite(cov)] < -1e-4).any()) or bool((cov[torch.isfinite(cov)] > 1 + 1e-4).any()):
                raise ValueError("[flow] covisibility must lie in [0, 1]")
            bad = self.valid & ~(torch.isfinite(self.flow).all(-1) & torch.isfinite(cov))
            if bool(bad.any()):
                raise ValueError("[flow] `valid` marks non-finite flow / covisibility")
            if not bool(self.valid.any()):
                raise ValueError("[flow] no valid pixel")
        return self


class FlowBackend(abc.ABC):
    """Base class of every flow / correspondence estimator."""

    #: registry key, e.g. "ufm"
    name: str = "base"
    #: True when ``predict`` is differentiable w.r.t. float tensor inputs
    supports_grad: bool = False
    #: True when the covisibility comes from the model itself (not from fwd/bwd consistency)
    native_covisibility: bool = False

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device)

    @abc.abstractmethod
    def predict(self, src: ImageLike, tgt: ImageLike) -> FlowOutput:
        """Flow from ``src`` to ``tgt`` (both (H,W,3) uint8 numpy or float tensors in [0,1])."""

    def __call__(self, src: ImageLike, tgt: ImageLike) -> FlowOutput:
        return self.predict(src, tgt)


# --------------------------------------------------------------------------- helpers
def image_to_bchw01(img: ImageLike, device: torch.device) -> torch.Tensor:
    """Convert an RGB image to a (1, 3, H, W) float32 tensor in [0, 1] on ``device``.

    Accepts uint8 / float numpy arrays (H,W,3) and uint8 / float tensors (H,W,3) or
    (3,H,W).  Float inputs are assumed to be in [0, 1] unless their max exceeds 1.5,
    in which case they are treated as [0, 255].  Autograd is preserved for tensors.
    """
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(img))
    elif torch.is_tensor(img):
        t = img
    else:
        raise TypeError(f"unsupported image type {type(img)}")
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError("flow backends take one image pair at a time")
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"image must be 3-D, got {tuple(t.shape)}")
    if t.shape[0] != 3 and t.shape[-1] == 3:  # HWC -> CHW
        t = t.permute(2, 0, 1)
    if t.shape[0] != 3:
        raise ValueError(f"image must have 3 channels, got {tuple(t.shape)}")
    t = t.to(device)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
        with torch.no_grad():
            if t.numel() and float(t.max()) > 1.5:
                t = t / 255.0
    return t.unsqueeze(0).contiguous()


def resize_to_max_side(
    x: torch.Tensor, max_side: Optional[int], multiple: int = 8
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Bilinearly shrink (1,3,H,W) so that its long side is <= ``max_side`` and both sides are multiples of ``multiple``.

    Returns the resized tensor and its (h, w).  Never upsamples.  ``max_side=None``
    keeps the resolution (only rounding to ``multiple`` if needed).
    """
    H, W = int(x.shape[-2]), int(x.shape[-1])
    scale = 1.0
    if max_side is not None and max(H, W) > max_side:
        scale = max_side / float(max(H, W))
    h = max(multiple, int(round(H * scale / multiple)) * multiple)
    w = max(multiple, int(round(W * scale / multiple)) * multiple)
    if (h, w) != (H, W):
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=scale < 1.0)
    return x, (h, w)


def rescale_flow(flow_b2hw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize a (B,2,h,w) flow field to (B,2,H,W) and scale the vectors accordingly."""
    h, w = int(flow_b2hw.shape[-2]), int(flow_b2hw.shape[-1])
    H, W = int(target_hw[0]), int(target_hw[1])
    if (h, w) == (H, W):
        return flow_b2hw
    out = F.interpolate(flow_b2hw, size=(H, W), mode="bilinear", align_corners=False)
    out = out.clone()
    out[:, 0] *= W / float(w)
    out[:, 1] *= H / float(h)
    return out


def rescale_map(x_b1hw: torch.Tensor, target_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """Resize a (B,C,h,w) scalar map to (B,C,H,W)."""
    if tuple(x_b1hw.shape[-2:]) == tuple(target_hw):
        return x_b1hw
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(x_b1hw, size=tuple(target_hw), mode=mode, **kwargs)


def make_flow_output(
    flow_b2hw: torch.Tensor,
    covisibility_bhw: torch.Tensor,
    extras: Optional[Dict[str, torch.Tensor]] = None,
) -> FlowOutput:
    """Package (1,2,H,W) flow and (1,H,W) covisibility into a validated :class:`FlowOutput`."""
    flow = flow_b2hw[0].permute(1, 2, 0).float()
    cov = covisibility_bhw[0].float()
    with torch.no_grad():
        valid = torch.isfinite(flow).all(-1) & torch.isfinite(cov)
    return FlowOutput(flow=flow, covisibility=cov, valid=valid, extras=dict(extras or {}))
