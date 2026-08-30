"""
GeCo: A Differentiable Geometric Consistency Metric for Video Generation.

This package hosts the pluggable model backends used by the metric:

* ``geco.geometry`` -- multi-view geometry estimators (camera intrinsics /
  extrinsics, per-frame depth and confidence).  Backends: ``vggt`` (default),
  ``vggt_omega``.
* ``geco.flow`` -- pairwise optical-flow / correspondence estimators that also
  provide a covisibility (non-occlusion) score.  Backends: ``ufm`` (default),
  ``sea_raft``, ``raft`` (torchvision), ``ptlflow`` (any model of the PTLFlow zoo).
* ``geco.metric`` -- the backend-agnostic per-pair error computation shared by
  ``demo_detection.py`` and ``GeCo-Eval_evaluation.py``.

Importing ``geco`` also registers the vendored third-party packages under
``external/`` on ``sys.path`` (VGGT, VGGT-Omega, UFM, SEA-RAFT) so that the
scripts work from a plain clone without a ``pip install -e .`` of every one of them.
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_ROOT = os.path.join(REPO_ROOT, "external")

# (sub-directory of external/, importable top-level package it provides)
_EXTERNAL_PACKAGES = (
    ("vggt", "vggt"),
    ("vggt_omega", "vggt_omega"),
    ("UFM", "uniflowmatch"),
    ("sea_raft", "sea_raft"),
)


def add_external_paths() -> None:
    """Put every vendored package root on ``sys.path`` (idempotent)."""
    for subdir, _pkg in _EXTERNAL_PACKAGES:
        path = os.path.join(EXTERNAL_ROOT, subdir)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


add_external_paths()

__version__ = "1.0.0"
__all__ = ["REPO_ROOT", "EXTERNAL_ROOT", "add_external_paths", "__version__"]
