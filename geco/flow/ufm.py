"""
UFM flow backend (Zhang et al., "UFM: A Simple Path towards Unified Dense Correspondence with Flow", 2025).

This is the correspondence estimator used for every number in the paper; it
ports ``utils.predict_correspondences`` one-to-one.

Native resolution
    UFM is trained at a fixed **560 x 420** (W x H) input.  Its
    ``predict_correspondences_batched`` resizes the pair to the trained
    resolution whose aspect ratio is closest to the input (``image_scaler``),
    runs the network there and maps the predicted flow back to the **input
    grid in input pixels** (``unmap_predicted_flow``), so the backend can feed
    frames at their source resolution (``max_side=None``, the default).  With
    ``max_side`` set, both frames are first shrunk so that their long side is
    ``<= max_side``; the flow (vectors rescaled) and covisibility are then
    resampled back onto the source grid.

Normalisation
    Frames are handed over as float32 RGB in ``[0, 1]`` with
    ``data_norm_type="identity"``; the conversion to the encoder's own
    normalisation (DINOv2 / ImageNet statistics) happens inside
    ``predict_correspondences_batched``.

Covisibility
    UFM has a dedicated *non-occluded mask* head: a sigmoid probability that the
    source pixel is visible in the target frame (neither occluded nor outside the
    target image).  It is returned as-is: 1 = covisible, 0 = occluded / out of view.

Differentiability
    The prediction is a pure tensor pipeline; when either input is a float tensor
    with ``requires_grad=True`` the forward pass runs with autograd enabled, so
    gradients flow back into the frames (used by test-time guidance).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

from .base import (
    FlowBackend,
    FlowOutput,
    ImageLike,
    image_to_bchw01,
    make_flow_output,
    rescale_flow,
    rescale_map,
    resize_to_max_side,
)

# variant -> (model class, default Hugging Face repo id)
_VARIANTS = {
    "base": (UniFlowMatchConfidence, "infinity1096/UFM-Base"),
    "refine": (UniFlowMatchClassificationRefinement, "infinity1096/UFM-Refine"),
}
DEFAULT_REPO_ID = _VARIANTS["base"][1]


class UFMFlow(FlowBackend):
    """UFM dense correspondence + native covisibility.

    Args:
        device:   torch device the model lives on.
        ckpt:     ``None`` -> the variant's default hub repo (``infinity1096/UFM-Base`` /
                  ``infinity1096/UFM-Refine``, cached under ``$HF_HOME``); a hub repo id or a
                  local directory in hub layout; or a ``.pt`` / ``.pth`` training checkpoint
                  (``{"model_args": ..., "model": state_dict}``).
        model:    ``None`` / ``"base"`` -> ``UniFlowMatchConfidence``;
                  ``"refine"`` -> ``UniFlowMatchClassificationRefinement``.
        max_side: optional cap on the long side fed to UFM (``None`` = source resolution).
    """

    name = "ufm"
    supports_grad = True
    native_covisibility = True

    def __init__(
        self,
        device: torch.device | str = "cuda",
        ckpt: Optional[str] = None,
        model: Optional[str] = None,
        max_side: Optional[int] = None,
    ):
        super().__init__(device=device)
        variant = "base" if model is None else str(model)
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown UFM variant {model!r}; choose one of {sorted(_VARIANTS)}")
        self.variant = variant
        self.max_side = None if max_side is None else int(max_side)

        cls, default_repo = _VARIANTS[variant]
        source = ckpt or default_repo
        if os.path.isfile(source):
            self.model = cls.from_pretrained_ckpt(source)
        else:
            self.model = cls.from_pretrained(source)
        self.model = self.model.to(self.device, dtype=torch.float32).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ helpers
    def model_hw(self, image_hw: Tuple[int, int]) -> Tuple[int, int]:
        """Network input (h, w) UFM picks for an ``image_hw`` pair (closest trained aspect ratio)."""
        H, W = image_hw
        candidates = [(int(res[1]), int(res[0])) for res in self.model.inference_resolution]  # (W, H) -> (H, W)
        return min(candidates, key=lambda hw: abs(hw[0] / hw[1] - H / W))

    # --------------------------------------------------------------- inference
    def predict(self, src: ImageLike, tgt: ImageLike) -> FlowOutput:
        src_t = image_to_bchw01(src, self.device)
        tgt_t = image_to_bchw01(tgt, self.device)
        if src_t.shape != tgt_t.shape:
            raise ValueError(f"source and target must share one resolution, got {tuple(src_t.shape)} vs {tuple(tgt_t.shape)}")
        H, W = int(src_t.shape[-2]), int(src_t.shape[-1])
        enable_grad = bool(src_t.requires_grad or tgt_t.requires_grad)

        with torch.set_grad_enabled(enable_grad):
            src_t, (h, w) = resize_to_max_side(src_t, self.max_side, multiple=1)
            tgt_t, _ = resize_to_max_side(tgt_t, self.max_side, multiple=1)
            out = self.model.predict_correspondences_batched(
                source_image=src_t.permute(0, 2, 3, 1).contiguous(),  # BHWC float32 in [0, 1]
                target_image=tgt_t.permute(0, 2, 3, 1).contiguous(),
                data_norm_type="identity",
            )
            if out.covisibility is None:
                raise RuntimeError("the loaded UFM checkpoint has no covisibility head")
            flow = out.flow.flow_output.float()  # (1, 2, h, w), pixels of the (h, w) grid
            cov = out.covisibility.mask.float()  # (1, h, w) in [0, 1]
            flow = rescale_flow(flow, (H, W))
            cov = rescale_map(cov[:, None], (H, W), mode="bilinear")[:, 0]

        return make_flow_output(flow, cov, extras={"model_hw": self.model_hw((h, w))}).validate(image_hw=(H, W))


__all__ = ["UFMFlow", "DEFAULT_REPO_ID"]
