"""
VGGT geometry backend (Wang et al., "Visual Geometry Grounded Transformer", CVPR 2025).

This is the estimator used for every number in the paper.  It reproduces the
original ``utils.vggt_infer`` + ``vggt.utils.load_fn.load_and_preprocess_images(mode="crop")``
pipeline, generalised so that portrait frames are resized like landscape ones
instead of being centre-cropped.

Native resolution
    VGGT-1B is a ViT with **patch size 14** trained at a long side of **518 px**.
    Frames are resized (bicubic from disk, bilinear + antialias for in-memory
    tensors) so that their long side equals ``long_side`` (518) and both sides
    are multiples of 14; the aspect ratio is preserved up to the rounding of the
    short side.  Nothing is ever cropped, so ``PreparedImages.crop_xywh`` is
    ``None`` and the whole output grid is valid.  Landscape frames go through
    exactly the same resize as ``load_and_preprocess_images(mode="crop")``.

Normalisation
    The network consumes RGB in ``[0, 1]``; the ImageNet mean / std
    normalisation is applied inside ``VGGT.aggregator``.

Outputs
    * ``intrinsic`` -- pinhole matrices in pixels of the output grid; VGGT
      predicts a field of view, so the principal point sits at the image centre.
    * ``extrinsic`` -- world-to-camera ``[R | t]`` (OpenCV convention).  The
      world frame is the **first camera** of the sequence.
    * ``depth`` -- z-depth in VGGT's (roughly metric) scale, bilinearly
      upsampled (``align_corners=False``) from the network resolution.
    * ``conf`` -- ``1 + exp(logit)``, i.e. ``>= 1`` everywhere, higher is better.
      Upsampled the same way as the depth.
    * ``extras["point_map"]`` (only with ``point_prediction=True``) -- the
      point-head world coordinates, (F, H, W, 3), upsampled like the depth.

Precision
    On CUDA the forward pass (aggregator *and* heads) runs under
    ``torch.autocast(compute_dtype)`` exactly like the paper code (bf16 on
    Ampere+).  The only difference: the predicted pose encoding is cast to
    float32 *before* it is turned into ``[R | t]`` / ``K`` -- a quaternion to
    matrix conversion carried out in bf16 yields rotations that are
    non-orthonormal by up to ~3 %, which :meth:`GeometryOutput.validate`
    rightly rejects.  Every output is returned as float32.
"""
from __future__ import annotations

import contextlib
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .base import (
    GeometryBackend,
    GeometryOutput,
    PreparedImages,
    crop_and_resize_frames,
    frames_to_fchw01,
    map_to_output_grid,
    resize_hw_to_multiple,
)

DEFAULT_REPO_ID = "facebook/VGGT-1B"
PATCH_SIZE = 14


class VGGTBackend(GeometryBackend):
    """VGGT-1B: cameras from the camera head, depth + confidence from the depth head.

    Args:
        device:           torch device the model lives on.
        ckpt:             ``None`` -> ``facebook/VGGT-1B`` from the Hugging Face hub
                          (cached under ``$HF_HOME``); a hub repo id; a local directory
                          in hub layout (``config.json`` + ``model.safetensors``); or a
                          ``.safetensors`` / ``.pt`` state-dict file.
        compute_dtype:    autocast dtype (default: bf16 on Ampere+, otherwise fp32 --
                          the paper-code rule; fp32 disables autocast).
        long_side:        long side of the network input, multiple of 14 (default 518).
        point_prediction: also run the point head and expose ``extras["point_map"]``.
    """

    name = "vggt"
    supports_grad = True

    def __init__(
        self,
        device: torch.device | str = "cuda",
        ckpt: Optional[str] = None,
        compute_dtype: Optional[torch.dtype] = None,
        long_side: int = 518,
        point_prediction: bool = False,
    ):
        super().__init__(device=device, compute_dtype=compute_dtype)
        if long_side % PATCH_SIZE != 0:
            raise ValueError(f"long_side={long_side} must be a multiple of the patch size {PATCH_SIZE}")
        self.long_side = int(long_side)
        self.point_prediction = bool(point_prediction)

        self.model = _load_model(ckpt, need_point_head=self.point_prediction).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ inputs
    def _target_hw(self, source_hw: Tuple[int, int]) -> Tuple[int, int]:
        return resize_hw_to_multiple(source_hw, self.long_side, PATCH_SIZE)

    def load_images(self, paths: Sequence[str]) -> PreparedImages:
        """Read frames like ``load_and_preprocess_images``: RGBA -> white background, bicubic resize."""
        if len(paths) == 0:
            raise ValueError("At least one frame is required")
        source_hw: Optional[Tuple[int, int]] = None
        target_hw: Tuple[int, int] = (0, 0)
        tensors = []
        for path in paths:
            img = Image.open(path)
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            img = img.convert("RGB")
            width, height = img.size
            if source_hw is None:
                source_hw = (height, width)
                target_hw = self._target_hw(source_hw)
            elif (height, width) != source_hw:
                raise ValueError(
                    f"All frames must share one resolution; {paths[0]} is {source_hw} but {path} is {(height, width)}"
                )
            img = img.resize((target_hw[1], target_hw[0]), Image.Resampling.BICUBIC)
            tensors.append(_pil_to_tensor01(img))
        images = torch.stack(tensors).to(self.device)
        return PreparedImages(images=images, source_hw=source_hw, crop_xywh=None)

    def preprocess(self, frames: torch.Tensor) -> PreparedImages:
        """Differentiable resize of in-memory frames (F,3,H,W) / (F,H,W,3) to the network grid."""
        frames = frames_to_fchw01(frames)
        source_hw = (int(frames.shape[-2]), int(frames.shape[-1]))
        images = crop_and_resize_frames(frames, None, self._target_hw(source_hw)).to(self.device)
        return PreparedImages(images=images, source_hw=source_hw, crop_xywh=None)

    # --------------------------------------------------------------- inference
    def _autocast(self):
        if self.compute_dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.compute_dtype)

    def infer(
        self,
        prepared: PreparedImages,
        output_hw: Optional[Tuple[int, int]] = None,
        enable_grad: bool = False,
    ) -> GeometryOutput:
        images = prepared.images.to(self.device)
        batch = images[None]  # (1, F, 3, h, w)
        point_map: Optional[torch.Tensor] = None

        with torch.set_grad_enabled(enable_grad):
            with self._autocast():
                tokens, patch_start_idx, _ = self.model.aggregator(batch)
                pose_enc = self.model.camera_head(tokens)[-1]  # (1, F, 9): T, quaternion, FoV
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.float(), prepared.model_hw)
                depth, conf, _ = self.model.depth_head(tokens, batch, patch_start_idx)
                if self.point_prediction:
                    point_map, _, _ = self.model.point_head(tokens, batch, patch_start_idx)

            # The upsample / intrinsics rescale stay inside the grad context so that
            # enable_grad=True yields differentiable outputs even under an ambient no_grad().
            depth = depth[0].float()  # (F, h, w, 1)
            conf = conf[0].float()[..., None]  # (F, h, w, 1)
            K, depth_out, conf_out, valid = map_to_output_grid(depth, conf, intrinsic[0].float(), prepared, output_hw)
            out_hw = (int(depth_out.shape[1]), int(depth_out.shape[2]))

            extras: Dict[str, torch.Tensor] = {}
            if point_map is not None:
                extras["point_map"] = _resize_hwc(point_map[0].float(), out_hw)

        return GeometryOutput(
            intrinsic=K.float(),
            extrinsic=extrinsic[0].float(),
            depth=depth_out.float(),
            conf=conf_out.float(),
            valid=valid,
            model_hw=prepared.model_hw,
            extras=extras,
        ).validate(image_hw=out_hw, num_frames=prepared.num_frames)


# --------------------------------------------------------------------------- helpers
def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """(3, h, w) float32 in [0, 1] -- identical to ``torchvision.transforms.ToTensor`` on an RGB image."""
    arr = np.array(img, dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)


def _resize_hwc(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear (align_corners=False) resize of a (F, h, w, C) map to (F, H, W, C)."""
    if (int(x.shape[1]), int(x.shape[2])) == tuple(hw):
        return x
    x = F.interpolate(x.permute(0, 3, 1, 2), size=tuple(hw), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Read a ``.safetensors`` or ``.pt`` state dict (optionally wrapped in a ``model`` / ``state_dict`` key)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a state dict")
    return obj


def _load_model(ckpt: Optional[str], need_point_head: bool) -> VGGT:
    """Build VGGT-1B from the hub (``ckpt=None`` / repo id), a hub-layout directory, or a state-dict file."""
    if ckpt is None:
        return VGGT.from_pretrained(DEFAULT_REPO_ID)
    ckpt = os.path.expanduser(ckpt)
    if os.path.isdir(ckpt):
        return VGGT.from_pretrained(ckpt)
    if os.path.isfile(ckpt):
        model = VGGT()
        missing, _unexpected = model.load_state_dict(_load_state_dict(ckpt), strict=False)
        required = ("aggregator.", "camera_head.", "depth_head.") + (("point_head.",) if need_point_head else ())
        bad = [k for k in missing if k.startswith(required)]
        if bad:
            raise RuntimeError(f"{ckpt} is missing {len(bad)} VGGT weights, e.g. {bad[:3]}")
        return model
    if ckpt.startswith((".", "/")) or ckpt.endswith((".pt", ".pth", ".safetensors")):
        raise FileNotFoundError(f"VGGT checkpoint not found: {ckpt}")
    return VGGT.from_pretrained(ckpt)  # Hugging Face repo id


__all__ = ["VGGTBackend", "DEFAULT_REPO_ID", "PATCH_SIZE"]
