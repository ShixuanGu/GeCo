"""
PTLFlow flow backend -- any two-frame model of the PTLFlow zoo (Morimitsu,
https://github.com/hmorimitsu/ptlflow) as a GeCo flow estimator.

PTLFlow is an **optional** dependency (``pip install ptlflow``); it is imported
lazily so that the rest of ``geco`` works without it.  The code targets the
PTLFlow 0.4.x API (``ptlflow.get_model(name, ckpt_path=...)``,
``IOAdapter(output_stride, input_size)`` / ``IOAdapter.unscale``); 0.3.x and
older expose ``IOAdapter(model, ...)`` / ``unpad_and_unscale`` instead and are
not supported.

Models and checkpoints
    ``model`` is a PTLFlow model name and ``ckpt`` one of that model's named
    checkpoints (downloaded by PTLFlow into ``$TORCH_HOME/hub/checkpoints``) or a
    local ``.ckpt`` file.  Combinations known to work on video frames:

    * ``dpflow`` (default) -- checkpoints ``things``, ``sintel`` (default),
      ``kitti``, ``spring``, ``chairs``.  DPFlow (CVPR 2025) adapts its pyramid
      depth to the input size, so it copes well with high-resolution frames.
    * ``sea_raft_s`` / ``sea_raft_m`` / ``sea_raft_l`` -- ``tartan``, ``chairs``,
      ``things``, ``sintel``, ``kitti``, ``spring`` (PTLFlow's re-implementation;
      the vendored ``sea_raft`` backend uses the authors' code and weights).
    * ``memflow`` -- ``things``, ``sintel``, ``kitti``, ``spring``; ``memflow_t``
      -- ``things``, ``things_kitti``, ``sintel``, ``kitti``.  MemFlow keeps a
      memory across calls; it is reset before every pair (``is_seq_start``)
      because the metric evaluates arbitrary frame pairs, not a video stream.

    ``videoflow_bof`` / ``videoflow_mof`` need three (or more) consecutive frames
    -- PTLFlow would replicate one frame of the pair and degrade silently -- so
    they are rejected here.  Any other two-frame model of the zoo should work
    but has not been checked.

Native resolution / normalisation
    PTLFlow models are fully convolutional and pad the input to their feature
    stride internally (``preprocess_images(resize_mode="pad")``), returning flow
    on the un-padded grid, so any resolution goes in.  Large frames are shrunk
    with ``resize_to_max_side`` until their long side is ``<= max_side`` (default
    1024) like the other RAFT-family backends and the outputs are resampled back
    to the source grid.  Following PTLFlow's ``cv2.imread`` convention the frames
    are handed over as **BGR** float32 in ``[0, 1]`` (HWC numpy through
    ``IOAdapter.prepare_inputs``); every model applies its own colour order and
    scaling (``bgr_to_rgb`` / ``bgr_add`` / ``bgr_mult``) inside ``forward``.
    Inference runs in float32 (autocast off).

Covisibility
    These are flow-only networks, so the forward (src -> tgt) and backward
    (tgt -> src) flows are predicted in two calls and combined with the
    forward-backward consistency check of :mod:`geco.flow.consistency`:
    ``cov > 0.5`` is exactly the classic Sundaram et al. "visible" decision and
    pixels whose match leaves the target image get ``cov = 0``.  The backward
    flow and the cycle error (both resampled to the source grid; the cycle-error
    values stay in pixels at the model's inference resolution) are exposed in
    ``extras``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import FlowBackend, FlowOutput, ImageLike, image_to_bchw01, make_flow_output, rescale_flow, rescale_map, resize_to_max_side
from .consistency import forward_backward_covisibility

DEFAULT_MODEL = "dpflow"
DEFAULT_CKPT = "sintel"
# models that consume more than two frames per prediction
_MULTI_FRAME_MODELS = ("videoflow_bof", "videoflow_mof")


class PTLFlowFlow(FlowBackend):
    """PTLFlow zoo model with forward-backward consistency as covisibility.

    Args:
        device:        torch device the model lives on.
        ckpt:          PTLFlow named checkpoint of the model (``None`` -> ``"sintel"``) or a local ``.ckpt`` file.
        model:         PTLFlow model name (``None`` -> ``"dpflow"``).
        max_side:      cap on the long side fed to the network (default 1024, ``None`` = no cap).
        model_kwargs:  constructor overrides of the PTLFlow model class (e.g. ``iters_per_level=8`` for
                       ``dpflow``); the checkpoint is then restored with ``ptlflow.restore_model``.
    """

    name = "ptlflow"
    supports_grad = False
    native_covisibility = False

    def __init__(
        self,
        device: torch.device | str = "cuda",
        ckpt: Optional[str] = None,
        model: Optional[str] = None,
        max_side: Optional[int] = 1024,
        **model_kwargs,
    ):
        try:
            import ptlflow
            from ptlflow.utils.io_adapter import IOAdapter
        except ImportError as e:
            raise ImportError(
                "The 'ptlflow' flow backend needs the optional PTLFlow package: pip install ptlflow"
            ) from e
        super().__init__(device=device)
        self.model_name = DEFAULT_MODEL if model is None else str(model)
        self.ckpt = DEFAULT_CKPT if ckpt is None else str(ckpt)
        if self.model_name in _MULTI_FRAME_MODELS:
            raise ValueError(
                f"PTLFlow model {self.model_name!r} needs three or more consecutive frames; "
                "GeCo evaluates frame pairs, so use a two-frame model (dpflow, sea_raft_m, memflow, ...)"
            )
        self.max_side = None if max_side is None else int(max_side)
        self._io_adapter_cls = IOAdapter

        if model_kwargs:
            net = ptlflow.restore_model(ptlflow.get_model_reference(self.model_name)(**model_kwargs), self.ckpt)
        else:
            net = ptlflow.get_model(self.model_name, ckpt_path=self.ckpt)
        self.model = net.to(self.device).eval()
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

            # the model pads to its own stride internally, so only the size cap is applied here
            src_t, (h, w) = resize_to_max_side(src_t, self.max_side, multiple=1)
            tgt_t, _ = resize_to_max_side(tgt_t, self.max_side, multiple=1)
            src_np, tgt_np = _to_bgr_hwc(src_t), _to_bgr_hwc(tgt_t)

            io_adapter = self._io_adapter_cls(output_stride=self.model.output_stride, input_size=(h, w), cuda=False)
            flow_fwd = self._flow(io_adapter, src_np, tgt_np)
            flow_bwd = self._flow(io_adapter, tgt_np, src_np)

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

    def _flow(self, io_adapter, img_a: np.ndarray, img_b: np.ndarray) -> torch.Tensor:
        """(1, 2, h, w) float32 flow from ``img_a`` to ``img_b`` (HWC BGR float32 numpy in [0, 1])."""
        inputs = io_adapter.prepare_inputs(images=[img_a, img_b])  # {"images": (1, 2, 3, h, w)}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["meta"] = {"is_seq_start": True}  # stateful models (MemFlow) must not carry memory across pairs
        preds = io_adapter.unscale(self.model(inputs))
        return preds["flows"][:, 0].float()  # (B, N-1, 2, h, w) -> (1, 2, h, w)


# --------------------------------------------------------------------------- helpers
def _to_bgr_hwc(x_13hw: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) RGB tensor in [0, 1] -> (H, W, 3) BGR float32 numpy array (PTLFlow's ``cv2.imread`` layout)."""
    return x_13hw[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)


__all__ = ["PTLFlowFlow", "DEFAULT_MODEL", "DEFAULT_CKPT"]
