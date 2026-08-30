"""
torchvision RAFT flow backend (Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms
for Optical Flow", ECCV 2020) -- an off-the-shelf alternative to UFM that needs no
vendored code: ``torchvision.models.optical_flow.raft_large`` / ``raft_small``.

Native resolution
    RAFT is fully convolutional; both sides of the input must be multiples of 8
    (features are extracted at 1/8 resolution).  It was trained on ~400 x 900
    crops (Sintel / KITTI), so very large frames are shrunk with
    ``resize_to_max_side`` until their long side is ``<= max_side`` (default 1024),
    rounded to multiples of 8.  The flow is bilinearly resampled back to the source
    grid with its vectors rescaled; the covisibility map is resampled the same way.
    torchvision's correlation pyramid needs feature maps of at least 16 x 16, i.e.
    both input sides must be **>= 128 px** after the optional shrink.

Weights
    ``ckpt=None`` selects torchvision's ``C_T_SKHT_V2`` (large) / ``C_T_V2`` (small)
    weights; any other member name of ``Raft_Large_Weights`` / ``Raft_Small_Weights``
    (e.g. ``"C_T_SKHT_K_V2"``, the KITTI fine-tune) or a local ``.pth`` state dict is
    accepted.  torchvision downloads the enum weights to ``$TORCH_HOME/hub/checkpoints``.

Normalisation
    Same as torchvision's ``OpticalFlow`` preset: RGB in ``[0, 1]`` mapped to
    ``[-1, 1]`` via ``(x - 0.5) / 0.5``.  Inference runs in float32 (autocast off).

Covisibility
    RAFT has no occlusion head, so the forward (src -> tgt) and backward
    (tgt -> src) flows are predicted in one batch of two and combined with the
    forward-backward consistency check of :mod:`geco.flow.consistency`:
    ``cov > 0.5`` is exactly the classic Sundaram et al. "visible" decision and
    pixels whose match leaves the target image get ``cov = 0``.  The backward flow
    and the cycle error (both resampled to the source grid; the cycle-error values
    stay in pixels at the model's inference resolution) are exposed in ``extras``.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

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
from .consistency import forward_backward_covisibility

# variant -> (builder, weights enum)
_VARIANTS = {
    "large": (raft_large, Raft_Large_Weights),
    "small": (raft_small, Raft_Small_Weights),
}
STRIDE = 8  # feature stride: input sides must be multiples of this


class TorchvisionRAFTFlow(FlowBackend):
    """RAFT (torchvision) with forward-backward consistency as covisibility.

    Args:
        device:   torch device the model lives on.
        ckpt:     ``None`` -> default torchvision weights (``C_T_SKHT_V2`` / ``C_T_V2``); a member
                  name of the variant's weights enum; or a local ``.pth`` state-dict file.
        model:    ``None`` / ``"large"`` -> ``raft_large``; ``"small"`` -> ``raft_small``.
        max_side: cap on the long side fed to RAFT (default 1024, ``None`` = no cap).
        iters:    number of recurrent flow updates (``num_flow_updates``, default 12).
    """

    name = "raft"
    supports_grad = False
    native_covisibility = False

    def __init__(
        self,
        device: torch.device | str = "cuda",
        ckpt: Optional[str] = None,
        model: Optional[str] = None,
        max_side: Optional[int] = 1024,
        iters: int = 12,
    ):
        super().__init__(device=device)
        variant = "large" if model is None else str(model)
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown RAFT variant {model!r}; choose one of {sorted(_VARIANTS)}")
        self.variant = variant
        self.max_side = None if max_side is None else int(max_side)
        self.iters = int(iters)

        builder, weights_enum = _VARIANTS[variant]
        if ckpt is not None and os.path.isfile(ckpt):
            self.model = builder(weights=None)
            self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        else:
            weights = weights_enum.DEFAULT if ckpt is None else _weights_by_name(weights_enum, ckpt)
            self.model = builder(weights=weights)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # --------------------------------------------------------------- inference
    def predict(self, src: ImageLike, tgt: ImageLike) -> FlowOutput:
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            src_t = image_to_bchw01(src, self.device)
            tgt_t = image_to_bchw01(tgt, self.device)
            if src_t.shape != tgt_t.shape:
                raise ValueError(
                    f"source and target must share one resolution, got {tuple(src_t.shape)} vs {tuple(tgt_t.shape)}"
                )
            H, W = int(src_t.shape[-2]), int(src_t.shape[-1])

            src_t, (h, w) = resize_to_max_side(src_t, self.max_side, multiple=STRIDE)
            tgt_t, _ = resize_to_max_side(tgt_t, self.max_side, multiple=STRIDE)
            if h % STRIDE or w % STRIDE:
                raise RuntimeError(f"RAFT input sides must be multiples of {STRIDE}, got {(h, w)}")

            # torchvision OpticalFlow preset: [0, 1] -> [-1, 1]
            pair_a = torch.cat([src_t, tgt_t], dim=0).sub_(0.5).div_(0.5)
            pair_b = torch.cat([tgt_t, src_t], dim=0).sub_(0.5).div_(0.5)
            flows = self.model(pair_a, pair_b, num_flow_updates=self.iters)[-1].float()  # (2, 2, h, w)
            flow_fwd, flow_bwd = flows[0:1], flows[1:2]

            cov, cycle_err, _in_bounds = forward_backward_covisibility(flow_fwd, flow_bwd)

            flow_fwd = rescale_flow(flow_fwd, (H, W))
            flow_bwd = rescale_flow(flow_bwd, (H, W))
            cov = rescale_map(cov[:, None], (H, W), mode="bilinear")[:, 0]
            cycle_err = rescale_map(cycle_err[:, None], (H, W), mode="bilinear")[:, 0]

        extras = {
            "flow_bwd": flow_bwd[0].permute(1, 2, 0).contiguous(),  # (H, W, 2) tgt -> src
            "cycle_error": cycle_err[0],  # (H, W) pixels
            "model_hw": (h, w),
        }
        return make_flow_output(flow_fwd, cov, extras=extras).validate(image_hw=(H, W))


# --------------------------------------------------------------------------- helpers
def _weights_by_name(weights_enum, name: str):
    """Resolve a weights enum member (e.g. ``"C_T_SKHT_K_V2"``) with a helpful error."""
    members = [m.name for m in weights_enum]
    if name not in members:
        raise ValueError(f"Unknown RAFT checkpoint {name!r}: expected a file path or one of {members}")
    return weights_enum[name]


__all__ = ["TorchvisionRAFTFlow", "STRIDE"]
