"""
Geometry backend contract.

Every geometry backend turns a list of video frames into per-frame camera
intrinsics / extrinsics, a z-depth map and a per-pixel confidence map, all
expressed on a common *output grid* (by default the resolution of the input
frames).  The metric code (``geco.metric``) only ever touches
:class:`GeometryOutput`, so adding a new estimator means implementing the three
abstract methods of :class:`GeometryBackend` and registering the class in
``geco/geometry/__init__.py``.

Conventions (identical to the original VGGT-based implementation)
-----------------------------------------------------------------
* ``intrinsic``: (F, 3, 3) float32 pinhole matrices **in pixels of the output
  grid** ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
* ``extrinsic``: (F, 3, 4) float32 world-to-camera ``[R | t]`` in the OpenCV
  camera convention (x right, y down, z forward).  The world frame is arbitrary
  (VGGT-family models anchor it to the first frame) -- only relative poses matter.
* ``depth``: (F, H, W, 1) float32 z-depth along the optical axis, > 0 where valid.
  The metric scale is backend dependent (the errors are normalised by depth).
* ``conf``: (F, H, W, 1) float32 confidence, *higher is better*.  The VGGT family
  emits ``1 + exp(logit)`` so values are ``>= 1``; other backends may use
  ``[0, 1]``.  ``--conf_min`` in the scripts is compared to this raw value.
* ``valid``: (F, H, W) bool -- pixels that carry a real prediction (finite depth
  > 0, finite confidence, inside the region the network actually saw).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def default_compute_dtype(device: torch.device) -> torch.dtype:
    """bf16 on Ampere+ GPUs (A100/H100), fp32 elsewhere -- same rule as the paper code."""
    if device.type == "cuda" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float32
    return torch.float32


@dataclass
class PreparedImages:
    """Frames after backend-specific preprocessing.

    ``images`` is the tensor fed to the network: (F, 3, h, w) float32 in [0, 1]
    (ImageNet normalisation, if any, is applied *inside* the backend).
    ``crop_xywh`` describes which region of the *source* frames ``images``
    depicts (x0, y0, width, height in source pixels); ``None`` means the full
    frame was only resized, never cropped.
    """

    images: torch.Tensor
    source_hw: Tuple[int, int]
    crop_xywh: Optional[Tuple[int, int, int, int]] = None

    @property
    def num_frames(self) -> int:
        return int(self.images.shape[0])

    @property
    def model_hw(self) -> Tuple[int, int]:
        return int(self.images.shape[-2]), int(self.images.shape[-1])


@dataclass
class GeometryOutput:
    intrinsic: torch.Tensor  # (F, 3, 3)
    extrinsic: torch.Tensor  # (F, 3, 4)
    depth: torch.Tensor  # (F, H, W, 1)
    conf: torch.Tensor  # (F, H, W, 1)
    valid: torch.Tensor  # (F, H, W) bool
    model_hw: Tuple[int, int]  # resolution the network ran at
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def num_frames(self) -> int:
        return int(self.depth.shape[0])

    @property
    def image_hw(self) -> Tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[2])

    def to(self, device) -> "GeometryOutput":
        return GeometryOutput(
            intrinsic=self.intrinsic.to(device),
            extrinsic=self.extrinsic.to(device),
            depth=self.depth.to(device),
            conf=self.conf.to(device),
            valid=self.valid.to(device),
            model_hw=self.model_hw,
            extras={k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.extras.items()},
        )

    def validate(
        self,
        image_hw: Optional[Tuple[int, int]] = None,
        num_frames: Optional[int] = None,
        rot_tol: float = 1e-2,
    ) -> "GeometryOutput":
        """Raise ``ValueError`` if the output violates the contract; returns ``self``."""
        Fn = int(self.depth.shape[0])
        if num_frames is not None and Fn != num_frames:
            raise ValueError(f"[geometry] expected {num_frames} frames, got {Fn}")
        if self.depth.ndim != 4 or self.depth.shape[-1] != 1:
            raise ValueError(f"[geometry] depth must be (F,H,W,1), got {tuple(self.depth.shape)}")
        H, W = int(self.depth.shape[1]), int(self.depth.shape[2])
        if image_hw is not None and (H, W) != tuple(image_hw):
            raise ValueError(f"[geometry] depth grid {(H, W)} != requested output grid {tuple(image_hw)}")
        if tuple(self.conf.shape) != (Fn, H, W, 1):
            raise ValueError(f"[geometry] conf must be {(Fn, H, W, 1)}, got {tuple(self.conf.shape)}")
        if tuple(self.valid.shape) != (Fn, H, W):
            raise ValueError(f"[geometry] valid must be {(Fn, H, W)}, got {tuple(self.valid.shape)}")
        if self.valid.dtype != torch.bool:
            raise ValueError(f"[geometry] valid must be bool, got {self.valid.dtype}")
        if tuple(self.intrinsic.shape) != (Fn, 3, 3):
            raise ValueError(f"[geometry] intrinsic must be {(Fn, 3, 3)}, got {tuple(self.intrinsic.shape)}")
        if tuple(self.extrinsic.shape) != (Fn, 3, 4):
            raise ValueError(f"[geometry] extrinsic must be {(Fn, 3, 4)}, got {tuple(self.extrinsic.shape)}")
        for name in ("intrinsic", "extrinsic", "depth", "conf"):
            t = getattr(self, name)
            if t.dtype != torch.float32:
                raise ValueError(f"[geometry] {name} must be float32, got {t.dtype}")
        devices = {self.intrinsic.device, self.extrinsic.device, self.depth.device, self.conf.device, self.valid.device}
        if len(devices) != 1:
            raise ValueError(f"[geometry] all tensors must live on one device, got {devices}")
        with torch.no_grad():
            if not torch.isfinite(self.intrinsic).all() or not torch.isfinite(self.extrinsic).all():
                raise ValueError("[geometry] non-finite camera parameters")
            fx, fy = self.intrinsic[:, 0, 0], self.intrinsic[:, 1, 1]
            if not bool((fx > 0).all() and (fy > 0).all()):
                raise ValueError("[geometry] focal lengths must be positive")
            if not torch.allclose(self.intrinsic[:, 2, 2], torch.ones_like(fx), atol=1e-4):
                raise ValueError("[geometry] intrinsic[2,2] must be 1")
            R = self.extrinsic[:, :, :3]
            RtR = R @ R.transpose(1, 2)
            eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(RtR)
            if (RtR - eye).abs().max().item() > rot_tol:
                raise ValueError("[geometry] extrinsic rotations are not orthonormal (is this really [R|t]?)")
            if not bool(self.valid.any()):
                raise ValueError("[geometry] no valid pixel in any frame")
            d = self.depth[..., 0]
            c = self.conf[..., 0]
            bad = self.valid & ~(torch.isfinite(d) & (d > 0) & torch.isfinite(c))
            if bool(bad.any()):
                raise ValueError("[geometry] `valid` marks pixels whose depth/conf are not finite & positive")
        return self


class GeometryBackend(abc.ABC):
    """Base class of every geometry estimator.

    Sub-classes must implement :meth:`load_images`, :meth:`preprocess` and
    :meth:`infer`.  Use :func:`map_to_output_grid` inside ``infer`` to turn the
    network-resolution predictions into the output grid; it handles the resize,
    the optional crop and the intrinsics bookkeeping uniformly for all backends.
    """

    #: registry key, e.g. "vggt"
    name: str = "base"
    #: True if ``infer(..., enable_grad=True)`` propagates gradients to the images
    supports_grad: bool = False

    def __init__(self, device: torch.device | str = "cuda", compute_dtype: Optional[torch.dtype] = None):
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype if compute_dtype is not None else default_compute_dtype(self.device)

    # ---------------------------------------------------------------- abstract
    @abc.abstractmethod
    def load_images(self, paths: Sequence[str]) -> PreparedImages:
        """Read frames from disk and resize them to the network resolution."""

    @abc.abstractmethod
    def preprocess(self, frames: torch.Tensor) -> PreparedImages:
        """Same as :meth:`load_images` but for in-memory frames.

        ``frames`` is (F, 3, H, W) or (F, H, W, 3), float in [0, 1] (or uint8),
        on any device.  Must be differentiable (pure tensor ops) so that
        test-time guidance can back-propagate into the frames.
        """

    @abc.abstractmethod
    def infer(
        self,
        prepared: PreparedImages,
        output_hw: Optional[Tuple[int, int]] = None,
        enable_grad: bool = False,
    ) -> GeometryOutput:
        """Run the network and express the results on ``output_hw`` (default: source resolution)."""

    # ------------------------------------------------------------- convenience
    def run(self, paths: Sequence[str], output_hw: Optional[Tuple[int, int]] = None) -> GeometryOutput:
        prepared = self.load_images(paths)
        return self.infer(prepared, output_hw=output_hw, enable_grad=False)

    def run_frames(self, frames: torch.Tensor, enable_grad: bool = False) -> GeometryOutput:
        prepared = self.preprocess(frames)
        return self.infer(prepared, output_hw=prepared.source_hw, enable_grad=enable_grad)

    def __call__(self, paths: Sequence[str], output_hw: Optional[Tuple[int, int]] = None) -> GeometryOutput:
        return self.run(paths, output_hw)


# --------------------------------------------------------------------------- utils
def resize_hw_to_multiple(hw: Tuple[int, int], long_side: int, multiple: int) -> Tuple[int, int]:
    """Target (h, w) whose *long* side equals ``long_side`` and whose sides are multiples of ``multiple``.

    ``long_side`` itself must be a multiple of ``multiple``.  Aspect ratio is preserved
    up to the rounding of the short side.
    """
    H, W = hw
    if long_side % multiple != 0:
        raise ValueError(f"long_side={long_side} must be a multiple of {multiple}")
    if W >= H:
        w = long_side
        h = max(multiple, int(round(H * (w / W) / multiple)) * multiple)
    else:
        h = long_side
        w = max(multiple, int(round(W * (h / H) / multiple)) * multiple)
    return h, w


def frames_to_fchw01(frames: torch.Tensor) -> torch.Tensor:
    """Accept (F,3,H,W) / (F,H,W,3) float [0,1] or uint8 and return (F,3,H,W) float32 in [0,1]."""
    if frames.ndim != 4:
        raise ValueError(f"frames must be 4-D, got {tuple(frames.shape)}")
    if frames.shape[-1] == 3 and frames.shape[1] != 3:
        frames = frames.permute(0, 3, 1, 2)
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    return frames.float()


def crop_and_resize_frames(
    frames: torch.Tensor,
    crop_xywh: Optional[Tuple[int, int, int, int]],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Differentiable crop + bilinear resize of (F,3,H,W) frames."""
    if crop_xywh is not None:
        x0, y0, w, h = crop_xywh
        frames = frames[:, :, y0 : y0 + h, x0 : x0 + w]
    if tuple(frames.shape[-2:]) != tuple(target_hw):
        frames = F.interpolate(frames, size=tuple(target_hw), mode="bilinear", align_corners=False, antialias=True)
    return frames


def map_to_output_grid(
    depth: torch.Tensor,
    conf: torch.Tensor,
    intrinsic: torch.Tensor,
    prepared: PreparedImages,
    output_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample network-resolution predictions onto the output grid.

    Args:
        depth:     (F, h, w, 1) z-depth at network resolution.
        conf:      (F, h, w, 1) confidence at network resolution.
        intrinsic: (F, 3, 3) intrinsics **in pixels of the (h, w) network input**.
        prepared:  the :class:`PreparedImages` the network consumed (carries the
                   source resolution and the optional crop window).
        output_hw: output grid; defaults to ``prepared.source_hw``.

    Returns:
        ``(intrinsic_out, depth_out, conf_out, valid)`` with shapes
        (F,3,3), (F,H,W,1), (F,H,W,1), (F,H,W).  Pixels outside the crop window
        (if any) get depth = conf = 0 and ``valid = False``.

    Without a crop this reduces exactly to the original ``vggt_infer`` behaviour:
    bilinear (align_corners=False) upsampling plus a per-axis focal / principal
    point rescale.
    """
    from utils import update_intrinsics  # local import: utils.py lives at the repo root

    Fn, h, w = int(depth.shape[0]), int(depth.shape[1]), int(depth.shape[2])
    Hs, Ws = prepared.source_hw
    H, W = (Hs, Ws) if output_hw is None else (int(output_hw[0]), int(output_hw[1]))
    sx, sy = W / float(Ws), H / float(Hs)

    if prepared.crop_xywh is None:
        x0, y0, Wc, Hc = 0, 0, Ws, Hs
    else:
        x0, y0, Wc, Hc = prepared.crop_xywh

    # crop window expressed on the output grid (integers)
    ox0 = int(round(x0 * sx))
    oy0 = int(round(y0 * sy))
    ox1 = min(W, int(round((x0 + Wc) * sx)))
    oy1 = min(H, int(round((y0 + Hc) * sy)))
    oW, oH = max(1, ox1 - ox0), max(1, oy1 - oy0)

    def _up(x: torch.Tensor) -> torch.Tensor:  # (F,h,w,1) -> (F,oH,oW,1)
        x = x.permute(0, 3, 1, 2).float()
        if (oH, oW) != (h, w):
            x = F.interpolate(x, size=(oH, oW), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1)

    depth_up = _up(depth)
    conf_up = _up(conf)

    K = update_intrinsics(intrinsic.float(), (h, w), (oH, oW))
    K = K.clone()
    K[:, 0, 2] += ox0
    K[:, 1, 2] += oy0

    if (ox0, oy0, oW, oH) == (0, 0, W, H):
        depth_out, conf_out = depth_up, conf_up
        region = torch.ones((Fn, H, W), dtype=torch.bool, device=depth.device)
    else:
        depth_out = depth_up.new_zeros((Fn, H, W, 1))
        conf_out = conf_up.new_zeros((Fn, H, W, 1))
        depth_out[:, oy0 : oy0 + oH, ox0 : ox0 + oW] = depth_up
        conf_out[:, oy0 : oy0 + oH, ox0 : ox0 + oW] = conf_up
        region = torch.zeros((Fn, H, W), dtype=torch.bool, device=depth.device)
        region[:, oy0 : oy0 + oH, ox0 : ox0 + oW] = True

    with torch.no_grad():
        d = depth_out[..., 0]
        c = conf_out[..., 0]
        valid = region & torch.isfinite(d) & (d > 0) & torch.isfinite(c)
    return K, depth_out, conf_out, valid
