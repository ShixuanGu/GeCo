"""
Shared command-line plumbing for the backend selection.

``demo_detection.py`` and ``GeCo-Eval_evaluation.py`` accept the same set of
backend flags; :func:`add_backend_args` registers them on an ``argparse`` parser,
:func:`build_backends` turns the parsed namespace into a ``(geometry, flow)``
pair of model objects and :func:`describe_backends` produces the one-line-per-
backend summary the scripts print at start-up.

>>> p = argparse.ArgumentParser()
>>> add_backend_args(p)
>>> args = p.parse_args(["--geometry", "vggt_omega", "--flow", "sea_raft"])
>>> geometry, flow = build_backends(args)
"""
from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple

import torch

from geco.flow import FlowBackend, build_flow, list_flow_backends
from geco.geometry import GeometryBackend, build_geometry, list_geometry_backends

DEFAULT_GEOMETRY = "vggt"
DEFAULT_FLOW = "ufm"

# Backend attributes worth showing in the start-up summary (only those a backend defines).
_INFO_ATTRS: Tuple[str, ...] = (
    "ckpt_path",
    "variant",
    "config",
    "model_name",
    "long_side",
    "image_resolution",
    "mode",
    "max_side",
    "iters",
    "supports_grad",
    "native_covisibility",
)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def add_backend_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register ``--geometry``, ``--flow`` and their options on ``parser``; returns the parser."""
    g = parser.add_argument_group("backends")
    g.add_argument(
        "--geometry",
        type=str,
        default=DEFAULT_GEOMETRY,
        choices=list_geometry_backends(),
        help="Multi-view geometry estimator (camera + depth + confidence).",
    )
    g.add_argument(
        "--geometry_ckpt",
        type=str,
        default=None,
        help="Checkpoint override for the geometry backend: local file / directory or Hugging Face repo id "
        "(default: the backend's official weights).",
    )
    g.add_argument(
        "--flow",
        type=str,
        default=DEFAULT_FLOW,
        choices=list_flow_backends(),
        help="Optical-flow / correspondence estimator (also provides the covisibility).",
    )
    g.add_argument(
        "--flow_model",
        type=str,
        default=None,
        help='Backend-specific model variant, e.g. SEA-RAFT config "spring-M", PTLFlow model "dpflow", '
        'UFM "base"/"refine", torchvision RAFT "large"/"small" (default: backend default).',
    )
    g.add_argument(
        "--flow_ckpt",
        type=str,
        default=None,
        help="Checkpoint override for the flow backend: local file, Hugging Face repo id, torchvision weight "
        'name or PTLFlow checkpoint name such as "sintel" (default: backend default).',
    )
    g.add_argument(
        "--flow_max_side",
        type=int,
        default=None,
        help="Run the flow model on frames downscaled to this long side (flow is rescaled back to the source "
        "resolution). 0 = no cap. Default: backend default (UFM = native resolution, RAFT family = 1024).",
    )
    g.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Torch device for both backends (default: cuda if available, else cpu).",
    )
    return parser


def build_backends(args: argparse.Namespace) -> Tuple[GeometryBackend, FlowBackend]:
    """Instantiate the geometry and flow backends selected by :func:`add_backend_args` flags.

    ``--flow_max_side`` is only forwarded when given so that every backend keeps
    its own default.  One summary line per backend is printed.
    """
    device = torch.device(args.device)
    geometry = build_geometry(args.geometry, device=device, ckpt=args.geometry_ckpt)
    print(describe_backend(geometry, ckpt=args.geometry_ckpt))

    flow_kwargs: dict[str, Any] = {}
    if args.flow_max_side is not None:
        flow_kwargs["max_side"] = None if args.flow_max_side <= 0 else int(args.flow_max_side)
    flow = build_flow(args.flow, device=device, ckpt=args.flow_ckpt, model=args.flow_model, **flow_kwargs)
    print(describe_backend(flow, ckpt=args.flow_ckpt))
    return geometry, flow


def backend_dtype(backend: Any) -> str:
    """Human-readable compute dtype: the backend's ``compute_dtype`` or, failing that, its parameter dtype."""
    dtype = getattr(backend, "compute_dtype", None)
    if dtype is None:
        model = getattr(backend, "model", None)
        if isinstance(model, torch.nn.Module):
            param = next(model.parameters(), None)
            dtype = None if param is None else param.dtype
    return "n/a" if dtype is None else str(dtype).replace("torch.", "")


def describe_backend(backend: Any, ckpt: Optional[str] = None) -> str:
    """One line: kind, name, checkpoint, device, dtype and the backend-specific settings it exposes."""
    kind = "geometry" if isinstance(backend, GeometryBackend) else "flow"
    ckpt_str = ckpt if ckpt is not None else getattr(backend, "ckpt_path", getattr(backend, "ckpt", None))
    parts = [
        f"[{kind}] {backend.name}",
        f"ckpt={'default' if ckpt_str is None else ckpt_str}",
        f"device={backend.device}",
        f"dtype={backend_dtype(backend)}",
    ]
    for attr in _INFO_ATTRS:
        if attr in ("ckpt_path", "ckpt") and ckpt_str is not None:
            continue  # already shown as ckpt=...
        value = getattr(backend, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    return " ".join(parts)


def describe_backends(
    geometry: GeometryBackend,
    flow: FlowBackend,
    geometry_ckpt: Optional[str] = None,
    flow_ckpt: Optional[str] = None,
) -> str:
    """Two-line summary of a backend pair for logs (see :func:`describe_backend`)."""
    return "\n".join((describe_backend(geometry, ckpt=geometry_ckpt), describe_backend(flow, ckpt=flow_ckpt)))


__all__ = [
    "DEFAULT_FLOW",
    "DEFAULT_GEOMETRY",
    "add_backend_args",
    "backend_dtype",
    "build_backends",
    "default_device",
    "describe_backend",
    "describe_backends",
]
