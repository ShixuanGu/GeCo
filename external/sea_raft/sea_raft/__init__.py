# BSD 3-Clause License
# Copyright (c) 2024, Princeton Vision & Learning Lab.  All rights reserved.
#
# Vendored from https://github.com/princeton-vl/SEA-RAFT -- see ../UPSTREAM.txt for
# the exact upstream commit and the list of local modifications, and ../LICENSE
# for the full license text.
"""
SEA-RAFT core (Wang, Lipson & Deng, "SEA-RAFT: Simple, Efficient, Accurate RAFT
for Optical Flow", ECCV 2024) as an importable package.

Upstream ships ``core/`` as a flat directory used through ``sys.path.append("core")``
plus JSON configs read by ``config/parser.py``.  This package keeps the network
code unchanged apart from package-relative imports and exposes what a caller needs:

* :class:`RAFT` -- the network: ``RAFT(args)`` with the :class:`argparse.Namespace`
  of a config; ``forward(img1, img2, iters=..., test_mode=True)`` takes RGB images
  in ``[0, 255]`` (it maps them to ``[-1, 1]`` itself) and returns a dict whose
  ``"flow"`` / ``"info"`` lists hold one (N, 2, H, W) flow and one (N, 4, H, W)
  uncertainty tensor per refinement iteration.
* :class:`InputPadder` -- pads a pair to multiples of 8 (``forward`` already
  applies it internally).
* :func:`load_config` / :func:`list_configs` -- the released ``config/eval/*.json``
  files (``spring-M``, ``sintel-M``, ``kitti-M`` and their ``-S`` / ``-L`` siblings)
  turned into the ``Namespace`` the model expects.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from .raft import RAFT
from .utils import InputPadder

#: directory holding the upstream ``config/eval/*.json`` files
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def list_configs() -> List[str]:
    """Sorted names of the bundled configs (file names without ``.json``)."""
    return sorted(os.path.splitext(f)[0] for f in os.listdir(CONFIG_DIR) if f.endswith(".json"))


def load_config(name_or_path: str) -> argparse.Namespace:
    """Load a SEA-RAFT config as an :class:`argparse.Namespace`.

    ``name_or_path`` is either a bundled config name (``"spring-M"``, see
    :func:`list_configs`) or the path of a JSON file.  Every key of the JSON
    becomes an attribute (the semantics of upstream ``config.parser.json_to_args``).
    """
    path = name_or_path if os.path.isfile(name_or_path) else os.path.join(CONFIG_DIR, f"{name_or_path}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Unknown SEA-RAFT config {name_or_path!r}: expected a JSON file path or one of {list_configs()}"
        )
    with open(path, "r") as f:
        data = json.load(f)
    return argparse.Namespace(**data)


__all__ = ["CONFIG_DIR", "InputPadder", "RAFT", "list_configs", "load_config"]
