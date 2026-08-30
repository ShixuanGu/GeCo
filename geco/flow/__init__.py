"""Optical-flow backend registry.

>>> from geco.flow import build_flow
>>> flow_model = build_flow("sea_raft", device="cuda")
>>> out = flow_model.predict(src_rgb_uint8, tgt_rgb_uint8)   # -> FlowOutput
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional

import torch

from .base import FlowBackend, FlowOutput, image_to_bchw01, make_flow_output, rescale_flow, rescale_map, resize_to_max_side

# name -> (module, class)
_REGISTRY: Dict[str, tuple] = {
    "ufm": ("geco.flow.ufm", "UFMFlow"),
    "sea_raft": ("geco.flow.sea_raft", "SEARAFTFlow"),
    "raft": ("geco.flow.raft_torchvision", "TorchvisionRAFTFlow"),
    "ptlflow": ("geco.flow.ptlflow_backend", "PTLFlowFlow"),
}

_INSTALL_HINTS = {
    "ufm": "UFM is vendored under external/UFM: `pip install -e external/UFM/UniCeption && pip install -e external/UFM`.",
    "sea_raft": "SEA-RAFT is vendored under external/sea_raft (no extra install); weights are fetched from the Hugging Face hub.",
    "raft": "torchvision >= 0.13 provides RAFT; weights are downloaded to $TORCH_HOME on first use.",
    "ptlflow": "Optional backend: `pip install ptlflow` (see README, 'Optional flow backends').",
}


def list_flow_backends() -> List[str]:
    return sorted(_REGISTRY.keys())


def register_flow_backend(name: str, module: str, cls: str) -> None:
    _REGISTRY[name] = (module, cls)


# Distinguishes "caller did not specify max_side" (keep the backend default: UFM native,
# RAFT family 1024) from an explicit ``max_side=None`` (no cap).
_UNSET = object()


def build_flow(
    name: str,
    device: torch.device | str = "cuda",
    ckpt: Optional[str] = None,
    model: Optional[str] = None,
    max_side=_UNSET,
    **kwargs,
) -> FlowBackend:
    """Instantiate a flow backend by registry name.

    Args:
        name:     registry key (``ufm``, ``sea_raft``, ``raft``, ``ptlflow``).
        device:   torch device.
        ckpt:     checkpoint path / Hugging Face id / named checkpoint (backend specific; ``None`` = default).
        model:    model variant (backend specific, e.g. SEA-RAFT config ``spring-M``, PTLFlow model ``dpflow``).
        max_side: cap on the long image side used for inference (flow is rescaled back to the
                  source resolution).  Leave unset to keep the backend's default (UFM: native
                  resolution, RAFT family: 1024); pass ``None`` for no cap.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown flow backend '{name}'. Available: {list_flow_backends()}")
    module_name, cls_name = _REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:  # pragma: no cover - depends on the environment
        hint = _INSTALL_HINTS.get(name, "")
        raise ImportError(f"Could not import flow backend '{name}' ({e}). {hint}") from e
    cls = getattr(module, cls_name)
    if max_side is not _UNSET:
        kwargs["max_side"] = max_side
    return cls(device=device, ckpt=ckpt, model=model, **kwargs)


__all__ = [
    "FlowBackend",
    "FlowOutput",
    "build_flow",
    "image_to_bchw01",
    "list_flow_backends",
    "make_flow_output",
    "register_flow_backend",
    "rescale_flow",
    "rescale_map",
    "resize_to_max_side",
]
