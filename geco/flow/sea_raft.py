"""
SEA-RAFT flow backend (Wang, Lipson & Deng, "SEA-RAFT: Simple, Efficient, Accurate
RAFT for Optical Flow", ECCV 2024).  The network is vendored under
``external/sea_raft`` (``import geco`` puts it on ``sys.path``); nothing to install.

Native resolution
    SEA-RAFT is fully convolutional with a 1/8 feature stride, so both input
    sides must be multiples of 8 (upstream pads with ``InputPadder``; we round
    the resize target instead so nothing has to be un-padded), and its 4-level
    correlation pyramid needs at least ``8 * 2**4 = 128`` px on each side
    (smaller frames are rejected -- the backend never upsamples).  The released
    Spring checkpoint was trained on 540 x 960 crops, hence large frames are
    shrunk with ``resize_to_max_side`` until their long side is ``<= max_side``
    (default 1024) and the flow / maps are resampled back to the source grid with
    the vectors rescaled.  The ``"scale"`` field of the upstream configs (a power
    of two by which ``custom.py`` resamples the pair before inference, ``-1`` =
    half resolution for Spring) is deliberately ignored: it records the
    resolution SEA-RAFT was evaluated at on one benchmark, whereas ``max_side``
    makes the inference resolution an explicit choice shared with the other
    RAFT-family backends.

Weights
    ``ckpt=None`` fetches the official Spring-M checkpoint
    ``MemorySlices/Tartan-C-T-TSKH-spring540x960-M`` (``model.safetensors``) from
    the Hugging Face hub (cached under ``$HF_HOME``); it pairs with the default
    config ``spring-M``.  ``ckpt`` may also be another hub repo id (again
    ``model.safetensors``) or a local ``.pth`` / ``.safetensors`` state dict.  For
    the other configs (``sintel-M``, ``kitti-M``, ``spring-L``, ... see
    ``sea_raft.list_configs()``) a matching checkpoint must be given explicitly.
    The ImageNet initialisation of the ResNet backbone is skipped
    (``args.init_weight = False``) because the checkpoint overwrites every
    parameter anyway.

Normalisation
    The network takes RGB in ``[0, 255]`` and maps it to ``[-1, 1]`` itself
    (``2 * (x / 255) - 1``).  Inference runs in float32 (autocast off).

Covisibility
    SEA-RAFT predicts flow plus a mixture-of-Laplacians uncertainty but no
    occlusion mask, so the forward (src -> tgt) and backward (tgt -> src) flows
    are computed in one batch of two and combined with the forward-backward
    consistency check of :mod:`geco.flow.consistency`: ``cov > 0.5`` is exactly
    the classic Sundaram et al. "visible" decision and pixels whose match leaves
    the target image get ``cov = 0``.  ``extras`` additionally exposes the
    backward flow, the cycle error and the forward uncertainty heat-map
    (``sum_k softmax(w)_k * log b_k`` over the two Laplacian components, exactly
    ``custom.get_heatmap`` upstream; larger = less certain), all resampled to the
    source grid (cycle-error values stay in pixels at the model's inference resolution).
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sea_raft import RAFT, load_config

from .base import FlowBackend, FlowOutput, ImageLike, image_to_bchw01, make_flow_output, rescale_flow, rescale_map, resize_to_max_side
from .consistency import forward_backward_covisibility

DEFAULT_CONFIG = "spring-M"
DEFAULT_REPO_ID = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
HUB_WEIGHTS_FILE = "model.safetensors"
STRIDE = 8  # feature stride: input sides must be multiples of this


class SEARAFTFlow(FlowBackend):
    """SEA-RAFT with forward-backward consistency as covisibility.

    Args:
        device:   torch device the model lives on.
        ckpt:     ``None`` -> ``MemorySlices/Tartan-C-T-TSKH-spring540x960-M`` (Spring-M); a hub repo id
                  providing ``model.safetensors``; or a local ``.pth`` / ``.safetensors`` state dict.
        model:    SEA-RAFT config name (``None`` -> ``"spring-M"``; see ``sea_raft.list_configs()``)
                  or the path of a config JSON.
        max_side: cap on the long side fed to the network (default 1024, ``None`` = no cap).
        iters:    number of recurrent refinement iterations (``None`` -> the config's ``iters``).
    """

    name = "sea_raft"
    supports_grad = False
    native_covisibility = False

    def __init__(
        self,
        device: torch.device | str = "cuda",
        ckpt: Optional[str] = None,
        model: Optional[str] = None,
        max_side: Optional[int] = 1024,
        iters: Optional[int] = None,
    ):
        super().__init__(device=device)
        self.config = DEFAULT_CONFIG if model is None else str(model)
        self.args = load_config(self.config)
        self.args.init_weight = False  # no ImageNet download: the checkpoint overwrites the backbone anyway
        self.max_side = None if max_side is None else int(max_side)
        self.iters = int(self.args.iters) if iters is None else int(iters)

        self.model = RAFT(self.args)  # also fills in args.corr_levels / corr_radius
        self.min_side = STRIDE * 2 ** int(self.args.corr_levels)  # smallest side the correlation pyramid supports
        load_weights(self.model, load_state_dict(DEFAULT_REPO_ID if ckpt is None else ckpt))
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
            if min(h, w) < self.min_side:
                raise ValueError(
                    f"SEA-RAFT needs frames of at least {self.min_side} px on each side, got {(h, w)} "
                    f"(source {(H, W)}, max_side={self.max_side})"
                )

            # one batch of two pairs, [src -> tgt, tgt -> src]; the network expects [0, 255]
            img1 = torch.cat([src_t, tgt_t], dim=0) * 255.0
            img2 = torch.cat([tgt_t, src_t], dim=0) * 255.0
            out = self.model(img1, img2, iters=self.iters, test_mode=True)
            flows = out["flow"][-1].float()  # (2, 2, h, w)
            info = out["info"][-1].float()  # (2, 4, h, w): two mixture logits + two raw log-scales
            flow_fwd, flow_bwd = flows[0:1], flows[1:2]
            uncertainty = uncertainty_heatmap(info[0:1], self.args.var_min, self.args.var_max)  # (1, h, w)

            cov, cycle_err, _in_bounds = forward_backward_covisibility(flow_fwd, flow_bwd)

            flow_fwd = rescale_flow(flow_fwd, (H, W))
            flow_bwd = rescale_flow(flow_bwd, (H, W))
            cov = rescale_map(cov[:, None], (H, W), mode="bilinear")[:, 0]
            cycle_err = rescale_map(cycle_err[:, None], (H, W), mode="bilinear")[:, 0]
            uncertainty = rescale_map(uncertainty[:, None], (H, W), mode="bilinear")[:, 0]

        extras = {
            "flow_bwd": flow_bwd[0].permute(1, 2, 0).contiguous(),  # (H, W, 2) tgt -> src
            "cycle_error": cycle_err[0],  # (H, W) pixels
            "uncertainty": uncertainty[0],  # (H, W) SEA-RAFT heat-map, larger = less certain
            "model_hw": (h, w),
        }
        return make_flow_output(flow_fwd, cov, extras=extras).validate(image_hw=(H, W))


# --------------------------------------------------------------------------- helpers
def uncertainty_heatmap(info_b4hw: torch.Tensor, var_min: float, var_max: float) -> torch.Tensor:
    """Per-pixel uncertainty from SEA-RAFT's ``info`` head (upstream ``custom.get_heatmap``).

    ``info`` holds, per pixel, the two mixture logits and the two raw log-scales
    of the mixture-of-Laplacians flow likelihood.  The "large" log-scale is
    clamped to ``[0, var_max]``, the "small" one to ``[var_min, 0]``, and the
    heat-map (B, H, W) is their softmax-weighted sum; larger = less certain.
    """
    weight = info_b4hw[:, :2].softmax(dim=1)
    log_b = torch.stack(
        [
            info_b4hw[:, 2].clamp(min=0.0, max=float(var_max)),
            info_b4hw[:, 3].clamp(min=float(var_min), max=0.0),
        ],
        dim=1,
    )
    return (weight * log_b).sum(dim=1)


def load_state_dict(source: str) -> Dict[str, torch.Tensor]:
    """State dict from a local ``.pth`` / ``.safetensors`` file or a hub repo id (``model.safetensors``), on CPU."""
    path = source if os.path.isfile(source) else hf_hub_download(repo_id=source, filename=HUB_WEIGHTS_FILE)
    if path.endswith(".safetensors"):
        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load ``state`` into ``model`` with strict key matching, except for alias keys.

    The hub ``.safetensors`` files drop the alias keys of shared modules
    (``BasicBlock.bn3`` is also registered as ``downsample.1``), so a plain
    ``strict=True`` load fails on the official checkpoint.  Instead the load runs
    with ``strict=False`` and then raises if any key is unexpected or if a missing
    key does not alias a tensor that was loaded (config / checkpoint mismatch).
    """
    result = model.load_state_dict(state, strict=False)
    tensors = model.state_dict(keep_vars=True)
    loaded = {id(tensors[k]) for k in tensors.keys() & state.keys()}
    missing = [k for k in result.missing_keys if id(tensors[k]) not in loaded]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "SEA-RAFT checkpoint does not match the config: "
            f"{len(missing)} tensors missing (e.g. {missing[:3]}), "
            f"{len(result.unexpected_keys)} unexpected (e.g. {result.unexpected_keys[:3]})"
        )


__all__ = ["SEARAFTFlow", "DEFAULT_CONFIG", "DEFAULT_REPO_ID", "STRIDE", "load_state_dict", "load_weights", "uncertainty_heatmap"]
