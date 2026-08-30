#!/usr/bin/env python3
"""GPU test: the `ufm` backend must reproduce the legacy `utils.predict_correspondences`, and the
RAFT-family backends must satisfy the flow contract on real frames.

Run on a GPU node:  python tests/test_flow_legacy_equivalence.py [--frames examples/deform_house] [--backends ufm,raft,sea_raft]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import geco  # noqa: F401
from geco.flow import build_flow
from geco.image_io import list_frame_files, load_image_rgb
from utils import predict_correspondences  # legacy path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="examples/deform_house")
    ap.add_argument("--backends", default="ufm,raft")
    ap.add_argument("--pair", default="0,4")
    args = ap.parse_args(argv)
    device = torch.device("cuda")
    files = list_frame_files(args.frames)
    i, j = (int(x) for x in args.pair.split(","))
    src, tgt = load_image_rgb(files[i]), load_image_rgb(files[j])
    H, W = src.shape[:2]
    print(f"pair ({i},{j}) {H}x{W}")

    for name in args.backends.split(","):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fb = build_flow(name, device=device)
        t_build = time.time() - t0
        t0 = time.time()
        out = fb.predict(src, tgt)
        torch.cuda.synchronize()
        t_pred = time.time() - t0
        mag = out.flow.norm(dim=-1)
        print(f"[{name}] build {t_build:.1f}s predict {t_pred:.2f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB | "
              f"flow |f| p50={mag.median():.2f} p95={mag.quantile(0.95):.2f} max={mag.max():.2f} px | "
              f"cov mean={out.covisibility.mean():.3f} frac>0.5={(out.covisibility > 0.5).float().mean():.3f} | "
              f"valid={out.valid.float().mean():.3f} extras={sorted(out.extras)} model_hw={out.extras.get('model_hw')}")
        if name == "ufm":
            with torch.no_grad():
                flow_l, cov_l = predict_correspondences(fb.model, src, tgt, "255")
            df = (out.flow - flow_l.float()).abs().max().item()
            dc = (out.covisibility - cov_l.float()).abs().max().item()
            print(f"[ufm] max |flow - legacy| = {df:.2e} px, max |cov - legacy| = {dc:.2e}")
            assert df < 1e-3 and dc < 1e-4, "ufm backend deviates from legacy predict_correspondences"
            # differentiable path
            s = torch.from_numpy(src).float().div(255).to(device).requires_grad_(True)
            t = torch.from_numpy(tgt).float().div(255).to(device)
            o = fb.predict(s, t)
            o.flow.norm(dim=-1).mean().backward()
            assert s.grad is not None and s.grad.abs().sum() > 0, "no gradient through UFM"
            print("[ufm] gradient w.r.t. source frame OK")
        del fb
        torch.cuda.empty_cache()
    print("PASS")


def test_flow_legacy_equivalence():
    """pytest entry point; skipped without CUDA (the test needs the flow weights + a GPU)."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("requires a CUDA GPU")
    main([])


if __name__ == "__main__":
    main()
