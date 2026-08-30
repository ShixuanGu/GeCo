"""
VGGT-Omega geometry backend (registry name ``"vggt_omega"``).

VGGT-Omega (Wang et al., 2026, https://github.com/facebookresearch/vggt-omega)
is the successor of VGGT: a 1B-parameter feed-forward multi-view transformer
that predicts, for every input frame, a 9-D camera encoding (translation, XYZW
quaternion, vertical / horizontal field of view), a z-depth map and a depth
confidence map.  The inference package is vendored under ``external/vggt_omega``
(``import geco`` puts it on ``sys.path``).

Input format
------------
* RGB frames in ``[0, 1]``, shape (F, 3, h, w).  The ImageNet mean / std
  normalisation is applied inside the aggregator, so the backend feeds raw
  ``[0, 1]`` values.  The patch size is 16, hence ``h`` and ``w`` are multiples of 16.
* The network resolution is selected by ``mode`` exactly as in the upstream
  ``vggt_omega.utils.load_fn.load_and_preprocess_images``:

  - ``"balanced"`` (default, the setting the released 512 checkpoint was tuned
    for) keeps the number of patch tokens close to ``(image_resolution / 16) ** 2``
    (1024 tokens for 512) while preserving the aspect ratio, e.g.
    1280x720 -> 688x384, 3:2 -> 624x416, square -> 512x512 (width x height).
  - ``"max_size"`` scales the longer side to ``image_resolution`` and rounds the
    other one to a multiple of 16, e.g. 1280x720 -> 512x288, 3:2 -> 512x336; it
    is cheaper (fewer tokens) but resolves less detail on wide frames.

* Aspect ratios (height / width) outside ``[0.5, 2]`` are first centre-cropped
  to the nearest bound (a 4:1 panorama keeps only its central 2:1 part).  When
  that happens the crop window is stored in ``PreparedImages.crop_xywh`` (source
  pixels) so that :func:`geco.geometry.base.map_to_output_grid` pastes the
  predictions back at the right place; source pixels outside the window get
  ``depth = conf = 0`` and ``valid = False``.  Frames of the common video aspect
  ratios (16:9, 4:3, 9:16, ...) are never cropped, only resized.
* File inputs are decoded with Pillow (RGBA composited onto white) and resized
  with bicubic filtering, bit-for-bit like upstream; in-memory frames go through
  the differentiable crop + antialiased bilinear resize of
  :func:`geco.geometry.base.crop_and_resize_frames` so that gradients reach the
  frames (``supports_grad = True``).

Outputs (see :mod:`geco.geometry.base` for the contract)
--------------------------------------------------------
* ``extrinsic``: world-to-camera OpenCV ``[R | t]``; the first frame defines the
  world frame.
* ``intrinsic``: pinhole matrix with the principal point at the image centre --
  VGGT-Omega only predicts the two fields of view -- expressed in pixels of the
  output grid.
* ``depth``: ``exp(logit) > 0`` z-depth in the model's own (up-to-scale) units.
* ``conf``: ``1 + exp(logit) >= 1``, higher is better.  This is the same
  "expp1" activation VGGT uses, so ``--conf_min`` / ``--conf_percentile`` behave
  identically for both backends.

Checkpoint and licence
----------------------
The weights (``vggt_omega_1b_512.pt``, ~4.6 GB, ``facebook/VGGT-Omega`` on the
Hugging Face Hub) are *gated*: request access at
https://huggingface.co/facebook/VGGT-Omega, then either log in
(``hf auth login`` or the ``HF_TOKEN`` environment variable) so that
``huggingface_hub`` can download them into ``$HF_HOME``, or download the file by
hand and pass its path (``--geometry_ckpt`` / ``--geometry-ckpt`` in the scripts).
The other official file, ``vggt_omega_1b_256_text.pt``
(``ckpt="facebook/VGGT-Omega:vggt_omega_1b_256_text.pt"``), carries an extra
text-alignment head that geometry estimation does not need; it is dropped at load
time, and the backend should then be built with ``image_resolution=256`` as
upstream recommends for that model.  Code and weights are distributed under the
FAIR Noncommercial Research License (``external/vggt_omega/LICENSE``); the GeCo
pipeline itself does not change that.

Upstream notice (August 2026): an ancestor of the released 1B checkpoint may
have suffered benchmark contamination, so the numbers reported for it on the
affected benchmarks may be inflated.  The model works as intended for downstream
use such as GeCo; simply do not use it to report results on those benchmarks.

Precision and device
--------------------
The backend calls the aggregator and the two heads directly rather than
``VGGTOmega.forward``, which hard-codes ``torch.autocast(device_type="cuda")``
(bf16 when supported, fp16 otherwise) and would neither honour ``compute_dtype``
nor run on CPU.  The numerics are the same: the aggregator runs under
``torch.autocast(compute_dtype)`` -- bf16 on Ampere+ by default, no autocast
otherwise (``compute_dtype=torch.float32``, the default off Ampere+) -- while the
camera / depth heads always run in float32 with autocast disabled, exactly like
upstream (``DenseHead`` insists on fp32 outputs, and a quaternion-to-matrix
conversion in bf16 yields non-orthonormal rotations).  ``compute_dtype=torch.float32``
is the choice for gradient-based guidance when mixed-precision gradients are too
noisy.  Every output is float32.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

from .base import (
    GeometryBackend,
    GeometryOutput,
    PreparedImages,
    crop_and_resize_frames,
    frames_to_fchw01,
    map_to_output_grid,
)

HF_REPO_ID = "facebook/VGGT-Omega"
HF_FILENAME = "vggt_omega_1b_512.pt"
HF_REQUEST_URL = "https://huggingface.co/facebook/VGGT-Omega"

PATCH_SIZE = 16
#: supported height / width range of the network input (upstream ``_crop_to_supported_aspect_ratio``)
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
MODES = ("balanced", "max_size")

# ``owner/repo`` optionally followed by ``:filename`` (no path separators in the file name)
_HUB_REF = re.compile(r"^[\w.-]+/[\w.-]+(?::[^/\\:]+)?$")


# ----------------------------------------------------------------------- preprocessing
def aspect_crop_window(source_hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Centre crop that brings ``height / width`` into ``[MIN_ASPECT_RATIO, MAX_ASPECT_RATIO]``.

    Mirrors ``vggt_omega.utils.load_fn._crop_to_supported_aspect_ratio``.  Returns
    ``(x0, y0, w, h)`` in source pixels, or ``None`` when the frame needs no crop.
    """
    H, W = int(source_hw[0]), int(source_hw[1])
    aspect = H / max(W, 1)
    if aspect < MIN_ASPECT_RATIO:  # too wide -> crop the width
        w = min(W, max(1, int(round(H / MIN_ASPECT_RATIO))))
        return max((W - w) // 2, 0), 0, w, H
    if aspect > MAX_ASPECT_RATIO:  # too tall -> crop the height
        h = min(H, max(1, int(round(W * MAX_ASPECT_RATIO))))
        return 0, max((H - h) // 2, 0), W, h
    return None


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(round(value / patch_size)) * patch_size)


def model_input_hw(
    cropped_hw: Tuple[int, int],
    image_resolution: int = 512,
    mode: str = "balanced",
    patch_size: int = PATCH_SIZE,
) -> Tuple[int, int]:
    """Network input (h, w) for a frame of (already aspect-cropped) size ``cropped_hw``.

    Mirrors ``_balanced_target_shape`` / ``_max_size_target_shape`` of the upstream
    loader: ``"balanced"`` keeps ~``(image_resolution / patch_size) ** 2`` patch
    tokens, ``"max_size"`` sets the longer side to ``image_resolution``.
    """
    H, W = int(cropped_hw[0]), int(cropped_hw[1])
    aspect = H / max(W, 1)
    if mode == "balanced":
        num_tokens = (image_resolution // patch_size) ** 2
        w_patches = math.sqrt(num_tokens / aspect)
        h_patches = num_tokens / w_patches
        return max(1, int(round(h_patches))) * patch_size, max(1, int(round(w_patches))) * patch_size
    if mode == "max_size":
        if aspect >= 1.0:
            return image_resolution, _round_to_patch_multiple(image_resolution / aspect, patch_size)
        return _round_to_patch_multiple(image_resolution * aspect, patch_size), image_resolution
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _load_rgb(path: str) -> Image.Image:
    """Decode an image as RGB, compositing RGBA inputs onto white (as upstream does)."""
    with Image.open(path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


# -------------------------------------------------------------------------- checkpoint
def resolve_checkpoint(ckpt: Optional[str]) -> str:
    """Return a local path to the VGGT-Omega weights, downloading from the Hub if needed.

    ``ckpt`` may be a local ``.pt`` file, ``None`` (default gated checkpoint
    ``facebook/VGGT-Omega`` / ``vggt_omega_1b_512.pt``), a Hub repo id
    (``"facebook/VGGT-Omega"``) or ``"repo_id:filename"``.  Downloads honour
    ``HF_HOME`` / ``HF_TOKEN`` like every other ``huggingface_hub`` call.
    """
    if ckpt is not None:
        ckpt = os.path.expanduser(ckpt)
        if os.path.isfile(ckpt):
            return ckpt
    if ckpt is None:
        repo_id, filename = HF_REPO_ID, HF_FILENAME
    elif _HUB_REF.match(ckpt):
        repo_id, _, filename = ckpt.partition(":")
        filename = filename or HF_FILENAME
    else:
        raise FileNotFoundError(
            f"VGGT-Omega checkpoint not found: {ckpt!r}. Pass a local .pt file, a Hub repo id "
            f"('{HF_REPO_ID}' or '{HF_REPO_ID}:{HF_FILENAME}'), or None for the default."
        )
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:  # gated / missing access, offline, no huggingface_hub, ...
        local_hint = f" (no local file named {ckpt!r} exists either)" if ckpt is not None else ""
        raise RuntimeError(
            f"Could not fetch '{filename}' from the gated Hugging Face repo '{repo_id}'{local_hint}: "
            f"{type(e).__name__}: {e}. Request access at {HF_REQUEST_URL}, then log in with "
            "`hf auth login` (or set HF_TOKEN), or download the file yourself and pass its local path "
            "via the scripts' --geometry_ckpt / --geometry-ckpt option."
        ) from e


# ----------------------------------------------------------------------------- backend
class VGGTOmegaBackend(GeometryBackend):
    """VGGT-Omega (1B, 512 px) camera + depth estimator.

    Args:
        device:           torch device the model lives on (CUDA or CPU).
        ckpt:             local path, Hub repo id or ``"repo_id:filename"``; ``None`` = gated default.
        compute_dtype:    autocast dtype of the aggregator (default: bf16 on Ampere+, otherwise
                          fp32 -- fp32 disables autocast); the heads always run in fp32.
        image_resolution: token budget / long side in pixels, a multiple of 16 (512 for the released
                          model, 256 for ``vggt_omega_1b_256_text.pt``).
        mode:             ``"balanced"`` (default) or ``"max_size"`` resolution policy.
    """

    name = "vggt_omega"
    supports_grad = True

    def __init__(
        self,
        device: torch.device | str = "cuda",
        ckpt: Optional[str] = None,
        compute_dtype: Optional[torch.dtype] = None,
        image_resolution: int = 512,
        mode: str = "balanced",
    ):
        super().__init__(device=device, compute_dtype=compute_dtype)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if image_resolution <= 0 or image_resolution % PATCH_SIZE != 0:
            raise ValueError(f"image_resolution must be a positive multiple of {PATCH_SIZE}, got {image_resolution}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested for VGGT-Omega but CUDA is not available")
        self.image_resolution = int(image_resolution)
        self.mode = mode
        self.ckpt_path = resolve_checkpoint(ckpt)
        ckpt_name = os.path.basename(self.ckpt_path)
        if "256" in ckpt_name and self.image_resolution != 256:
            warnings.warn(
                f"checkpoint {ckpt_name!r} looks like a 256-px VGGT-Omega model but "
                f"image_resolution={self.image_resolution}; build the backend with "
                "image_resolution=256 (see the module docstring)."
            )

        model = VGGTOmega(patch_size=PATCH_SIZE, enable_alignment=False)
        state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        # the 256-px text-aligned checkpoint carries a text-alignment head that geometry does not use
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("text_alignment_head.")}
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)

    # ------------------------------------------------------------------ helpers
    def _network_hw(self, source_hw: Tuple[int, int], crop_xywh: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int]:
        cropped_hw = (crop_xywh[3], crop_xywh[2]) if crop_xywh is not None else source_hw
        return model_input_hw(cropped_hw, self.image_resolution, self.mode, PATCH_SIZE)

    def _autocast(self):
        """Mixed-precision context of the aggregator (no autocast for float32)."""
        if self.compute_dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.compute_dtype)

    # ----------------------------------------------------------------- contract
    def load_images(self, paths: Sequence[str]) -> PreparedImages:
        """Decode, aspect-crop and bicubically resize frames exactly like the upstream loader."""
        if len(paths) == 0:
            raise ValueError("At least one frame is required")
        images = []
        source_hw: Optional[Tuple[int, int]] = None
        crop_xywh: Optional[Tuple[int, int, int, int]] = None
        target_hw: Tuple[int, int] = (0, 0)
        for path in paths:
            image = _load_rgb(path)
            W, H = image.size
            if source_hw is None:
                source_hw = (H, W)
                crop_xywh = aspect_crop_window(source_hw)
                target_hw = self._network_hw(source_hw, crop_xywh)
            elif (H, W) != source_hw:
                raise ValueError(f"All frames must share one resolution; {paths[0]} is {source_hw} but {path} is {(H, W)}")
            if crop_xywh is not None:
                x0, y0, w, h = crop_xywh
                image = image.crop((x0, y0, x0 + w, y0 + h))
            image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.BICUBIC)
            images.append(torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1))
        tensor = torch.stack(images).float().div_(255.0)
        return PreparedImages(images=tensor, source_hw=source_hw, crop_xywh=crop_xywh)

    def preprocess(self, frames: torch.Tensor) -> PreparedImages:
        """Differentiable counterpart of :meth:`load_images` for in-memory frames."""
        frames = frames_to_fchw01(frames)
        source_hw = (int(frames.shape[-2]), int(frames.shape[-1]))
        crop_xywh = aspect_crop_window(source_hw)
        target_hw = self._network_hw(source_hw, crop_xywh)
        images = crop_and_resize_frames(frames, crop_xywh, target_hw)
        return PreparedImages(images=images, source_hw=source_hw, crop_xywh=crop_xywh)

    def infer(
        self,
        prepared: PreparedImages,
        output_hw: Optional[Tuple[int, int]] = None,
        enable_grad: bool = False,
    ) -> GeometryOutput:
        h, w = prepared.model_hw
        out_hw = prepared.source_hw if output_hw is None else (int(output_hw[0]), int(output_hw[1]))
        batch = prepared.images.to(self.device, dtype=torch.float32, non_blocking=True)[None]  # (1, F, 3, h, w)

        # same schedule as VGGTOmega.forward: aggregator under autocast, heads in float32
        with torch.set_grad_enabled(enable_grad):
            with self._autocast():
                tokens, patch_token_start = self.model.aggregator(batch)
            with torch.autocast(device_type=self.device.type, enabled=False):
                pose_enc = self.model.camera_head(tokens, patch_token_start=patch_token_start)  # (1, F, 9)
                extrinsic, intrinsic = encoding_to_camera(pose_enc.float(), (h, w))
                depth, conf = self.model.dense_head(tokens, images=batch, patch_token_start=patch_token_start)
            extrinsic = extrinsic[0].float()  # (F, 3, 4) world -> camera, OpenCV
            intrinsic = intrinsic[0].float()  # (F, 3, 3) in pixels of the (h, w) network input
            depth = depth[0].float()  # (F, h, w, 1)
            conf = conf[0].float()[..., None]  # (F, h, w, 1), 1 + exp(logit)
            K, depth_out, conf_out, valid = map_to_output_grid(depth, conf, intrinsic, prepared, out_hw)

        return GeometryOutput(
            intrinsic=K,
            extrinsic=extrinsic,
            depth=depth_out,
            conf=conf_out,
            valid=valid,
            model_hw=(h, w),
        ).validate(image_hw=out_hw, num_frames=prepared.num_frames)


__all__ = ["VGGTOmegaBackend", "aspect_crop_window", "model_input_hw", "resolve_checkpoint"]
