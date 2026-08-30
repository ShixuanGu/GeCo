#!/usr/bin/env python3
"""GPU test: the `vggt` backend must reproduce the legacy `utils.vggt_infer` pipeline bit-for-bit
(up to autocast non-determinism) on landscape frames.

Run on a GPU node:  python tests/test_vggt_legacy_equivalence.py [--frames examples/deform_house] [--num_frames 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import geco  # noqa: F401
from geco.geometry import build_geometry
from geco.image_io import list_frame_files, image_hw
from utils import vggt_infer  # legacy path
from vggt.utils.load_fn import load_and_preprocess_images


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="examples/deform_house")
    ap.add_argument("--num_frames", type=int, default=8)
    args = ap.parse_args(argv)

    device = torch.device("cuda")
    files = list_frame_files(args.frames)[: args.num_frames]
    H, W = image_hw(files[0])
    torch.manual_seed(0)

    backend = build_geometry("vggt", device=device)
    dtype = backend.compute_dtype
    print(f"frames={len(files)} source={H}x{W} compute_dtype={dtype}")

    # --- legacy -------------------------------------------------------------
    t0 = time.time()
    imgs = load_and_preprocess_images(files, "crop", patch_size=14).to(device)[None]
    with torch.no_grad():
        legacy = vggt_infer(backend.model, imgs, (H, W), False, dtype, device)
    torch.cuda.synchronize()
    t_legacy = time.time() - t0

    # --- backend ------------------------------------------------------------
    t0 = time.time()
    prep = backend.load_images(files)
    geo = backend.infer(prep, output_hw=(H, W))
    torch.cuda.synchronize()
    t_new = time.time() - t0

    assert tuple(prep.images.shape[-2:]) == tuple(imgs.shape[-2:]), (prep.images.shape, imgs.shape)
    img_diff = (prep.images - imgs[0]).abs().max().item()
    print(f"model input {tuple(prep.images.shape)}  max|img_new-img_legacy|={img_diff:.2e}")

    def rel(a, b):
        return ((a - b).abs().max() / (b.abs().max() + 1e-8)).item()

    K_d = rel(geo.intrinsic, legacy["intrinsic"].float())
    E_d = rel(geo.extrinsic, legacy["extrinsic"].float())
    D_d = rel(geo.depth, legacy["depth_map"].float())
    C_d = rel(geo.conf, legacy["vggt_conf"].float())
    print(f"max rel diff  K={K_d:.2e}  E={E_d:.2e}  depth={D_d:.2e}  conf={C_d:.2e}")
    print(f"depth range new [{geo.depth[geo.valid].min():.3f}, {geo.depth[geo.valid].max():.3f}]  "
          f"legacy [{legacy['depth_map'].min():.3f}, {legacy['depth_map'].max():.3f}]")
    print(f"K[0]=\n{geo.intrinsic[0].cpu().numpy()}")
    print(f"time legacy {t_legacy:.2f}s  new {t_new:.2f}s  peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    # Same inputs -> same outputs, modulo autocast (bf16) non-determinism of the attention kernels.
    tol = 5e-2 if dtype != torch.float32 else 1e-5
    assert img_diff < 1e-6, "preprocessing differs from load_and_preprocess_images(mode='crop')"
    assert K_d < tol and E_d < tol and D_d < tol and C_d < tol, "backend deviates from the legacy pipeline"
    print("PASS: vggt backend == legacy vggt_infer pipeline")


def test_vggt_legacy_equivalence():
    """pytest entry point; skipped without CUDA (the test needs the VGGT weights + a GPU)."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("requires a CUDA GPU")
    main([])


if __name__ == "__main__":
    main()
