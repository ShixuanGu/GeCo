"""Geometry backend registry.

>>> from geco.geometry import build_geometry
>>> geo_model = build_geometry("vggt_omega", device="cuda", ckpt="/path/to/vggt_omega_1b_512.pt")
>>> geo = geo_model.run(frame_paths)          # -> GeometryOutput at the frames' native resolution
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional

import torch

from .base import GeometryBackend, GeometryOutput, PreparedImages, default_compute_dtype, map_to_output_grid

# name -> (module, class)
_REGISTRY: Dict[str, tuple] = {
    "vggt": ("geco.geometry.vggt", "VGGTBackend"),
    "vggt_omega": ("geco.geometry.vggt_omega", "VGGTOmegaBackend"),
}

# Optional third-party dependency hints shown when an import fails.
_INSTALL_HINTS = {
    "vggt": "VGGT is vendored under external/vggt (pip install -r requirements.txt).",
    "vggt_omega": "VGGT-Omega is vendored under external/vggt_omega; request the checkpoint at "
    "https://huggingface.co/facebook/VGGT-Omega and pass --geometry_ckpt (or log in with `hf auth login`).",
}


def list_geometry_backends() -> List[str]:
    return sorted(_REGISTRY.keys())


def register_geometry_backend(name: str, module: str, cls: str) -> None:
    _REGISTRY[name] = (module, cls)


def build_geometry(
    name: str,
    device: torch.device | str = "cuda",
    ckpt: Optional[str] = None,
    compute_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> GeometryBackend:
    """Instantiate a geometry backend by registry name.

    ``ckpt`` is a local path or a Hugging Face repo id (backend specific default
    when ``None``).  Extra ``kwargs`` are forwarded to the backend constructor.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown geometry backend '{name}'. Available: {list_geometry_backends()}")
    module_name, cls_name = _REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:  # pragma: no cover - depends on the environment
        hint = _INSTALL_HINTS.get(name, "")
        raise ImportError(f"Could not import geometry backend '{name}' ({e}). {hint}") from e
    cls = getattr(module, cls_name)
    return cls(device=device, ckpt=ckpt, compute_dtype=compute_dtype, **kwargs)


__all__ = [
    "GeometryBackend",
    "GeometryOutput",
    "PreparedImages",
    "build_geometry",
    "default_compute_dtype",
    "list_geometry_backends",
    "map_to_output_grid",
    "register_geometry_backend",
]
