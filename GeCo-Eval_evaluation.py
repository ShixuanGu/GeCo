#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Benchmark Evaluator for Video Consistency.

This script iterates over a structured dataset of video frames (Model -> Category -> Clip),
computes consistency scores with a geometry backend (structure) and a flow backend (motion),
and aggregates results.  Backends are chosen with the shared ``--geometry`` / ``--flow``
flags (``geco.cli``); the paper numbers use the defaults (VGGT + UFM).

Outputs:
1. Per-clip CSVs: Saved as `<csv_dir>/<model_key>.csv` (clip_id, category, motion, depth, fused).
2. Summary: Prints aggregated mean scores per category to stdout.

Sampling Strategy:
Instead of processing every frame (which is slow), this script divides the video into
a fixed number of temporal windows (e.g., 4 windows of 3 seconds each) and computes
the score within those windows.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, total=None, desc=None, unit=None): return it

import geco  # noqa: F401  (puts the vendored third-party packages on sys.path)
from geco.cli import add_backend_args, build_backends
from geco.flow import FlowBackend
from geco.geometry import GeometryBackend
from geco.image_io import check_same_size, list_frame_files, load_image_rgb
from geco.metric import ErrorAccumulator, compute_pair_errors, source_confidence_mask
from utils import build_fixed_len_cover_windows

# ---------------------- Configuration ----------------------

# Mapping of Model Name -> Native FPS
# Used to determine time-based sampling
dic_model_fps = {
    'Gen_CogVideoX_2b': 16,
    'Gen_CogVideoX_5b': 16,
    'Gen_CogVideoX1.5_5b': 16,
    'Gen_LTX': 30,
    'Gen_SORA2': 30,
    'Gen_Veo3.1': 24,
    'Gen_WAN2.2': 16,
    'Gen_HunyuanVideo': 24,
    'Gen_SORA2_480p': 30,
    'Gen_Veo3.1_480p': 24,
    'Gen_WAN2.2_480p': 16,
    'Gen_CogVideoX1.5_5b_480p': 16,
}

ALL_CATEGORIES = [
    "indoor_prompts",
    "object_centric_prompts",
    "outdoor_prompts",
    "stress_test_prompts",
]

# ---------------------- Evaluation Logic ----------------------

@torch.no_grad()
def evaluate_one_window(
    image_files: List[str],
    geometry: GeometryBackend,
    flow: FlowBackend,
    args: argparse.Namespace,
) -> Tuple[float, float, float, int]:
    """
    Evaluates consistency for a single window of frames.

    Steps:
    1. Run the geometry backend once on the whole window to get Camera Poses & Depth.
    2. Iterate through pairs (Source -> Target).
    3. Compute Motion Error (Flow vs Ego-motion).
    4. Compute Depth Error (Unidirectional Reprojection).
    5. Fuse errors based on occlusion logic.
    (Steps 3-5 live in ``geco.metric.compute_pair_errors``.)
    """
    H, W = check_same_size(image_files)

    # Cache the frames to reduce disk I/O latency
    cached_imgs = [load_image_rgb(fp) for fp in image_files]

    # 1. Geometry Inference (Batch) on the frames' native grid
    geo = geometry.run(image_files, output_hw=(H, W))

    N = len(image_files)
    per_frame_motion, per_frame_depth, per_frame_fused = [], [], []

    step = max(1, int(args.pair_stride))

    for src in range(N):
        src_img = cached_imgs[src]

        # Accumulators
        acc = ErrorAccumulator((H, W), geo.depth.device)

        # Source Confidence Mask
        conf_src_mask = source_confidence_mask(geo, src, args.conf_percentile, args.conf_min)

        for tgt in range(0, N, step):
            if tgt == src:
                continue

            # 2. Flow & Covisibility
            flow_out = flow.predict(src_img, cached_imgs[tgt])

            # 3-5. Motion residual, unidirectional depth consistency, fusion
            pe = compute_pair_errors(
                geo, src, tgt, flow_out, conf_src_mask,
                covis_thresh=args.covis_thresh, tau_z=args.tau_z, depth_check="unidirectional",
            )
            acc.add(pe)

        # Compute scalars for this frame (masked mean over pixels with >= 1 contribution)
        motion_mean, depth_mean, fused_mean = acc.scalars()
        per_frame_motion.append(motion_mean)
        per_frame_depth.append(depth_mean)
        per_frame_fused.append(fused_mean)

    # Window Averaging
    def nanmean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)) if arr.size > 0 else float('nan')

    return nanmean(per_frame_motion), nanmean(per_frame_depth), nanmean(per_frame_fused), len(image_files)


def evaluate_video_folder(
    video_dir: str,
    model_key: str,
    geometry: GeometryBackend,
    flow: FlowBackend,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    """Entry point for a single video folder."""
    all_files = list_frame_files(video_dir)
    if len(all_files) < 2:
        return float('nan'), float('nan'), float('nan')

    # Get Sampling Windows
    if model_key in dic_model_fps:
        fps_native = float(dic_model_fps[model_key])
    elif args.default_fps is not None:
        fps_native = float(args.default_fps)
    else:
        raise KeyError(
            f"Unknown model folder {model_key!r}: add it to dic_model_fps or pass --default_fps "
            f"(known models: {sorted(dic_model_fps)})"
        )
    eval_fps_eff = min(float(args.eval_fps), fps_native)

    windows, _ = build_fixed_len_cover_windows(
        all_files=all_files,
        fps_native=fps_native,
        eval_fps=eval_fps_eff,
        win_sec=args.win_sec,
        max_windows=args.max_windows
    )

    if not windows:
        return float('nan'), float('nan'), float('nan')

    # Run windows
    motion_vals, depth_vals, fused_vals, weights = [], [], [], []
    for wfiles in windows:
        m, d, f, w = evaluate_one_window(wfiles, geometry, flow, args)
        if np.isfinite(m) and np.isfinite(d) and np.isfinite(f) and w >= 2:
            motion_vals.append(m)
            depth_vals.append(d)
            fused_vals.append(f)
            weights.append(w)

    if not weights:
        return float('nan'), float('nan'), float('nan')

    # Weighted Average across windows
    W = np.array(weights, dtype=float)
    W = W / W.sum()

    motion_score = float((np.array(motion_vals) * W).sum())
    depth_score  = float((np.array(depth_vals)  * W).sum())
    fused_score  = float((np.array(fused_vals)  * W).sum())
    return motion_score, depth_score, fused_score


# ---------------------- Helper: Filter Directories ----------------------
def pick_suffix_subdirs(cat_dir: str, suffixes: List[str] | None) -> List[str]:
    """Returns subdirectories ending with specific suffixes (e.g., '_1')."""
    if not os.path.isdir(cat_dir):
        return []
    subs = [d for d in sorted(os.listdir(cat_dir)) if os.path.isdir(os.path.join(cat_dir, d))]

    if suffixes is None:
        return subs

    keep = []
    suffixes = [str(s) for s in suffixes]
    for d in subs:
        for s in suffixes:
            if d.endswith(f"_{s}"):
                keep.append(d)
                break
    return keep

# ---------------------- Main Execution ----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluate video consistency across models and categories."
    )
    p.add_argument("--frames_root", type=str, required=True,
                   help="Root folder containing model subfolders (Gen_*).")
    p.add_argument("--models", type=str, nargs="+", required=True,
                   help="Model keys (e.g. Gen_SORA2) or 'all'.")
    p.add_argument("--categories", type=str, nargs="+", default=ALL_CATEGORIES,
                   choices=ALL_CATEGORIES, help="Categories to evaluate.")
    p.add_argument("--suffixes", type=str, nargs="+", default=["all"],
                   help="Filter clip directories by suffix (e.g. '1', '2' or 'all').")

    # Sampling
    p.add_argument("--win_sec", type=float, default=3.0, help="Window duration (sec).")
    p.add_argument("--max_windows", type=int, default=4, help="Max windows per clip.")
    p.add_argument("--eval_fps", type=float, default=8.0, help="Sampling FPS.")
    p.add_argument("--default_fps", type=float, default=None,
                   help="Native FPS assumed for model folders not listed in dic_model_fps "
                        "(lets you benchmark your own model without editing the source).")

    # Algorithm
    p.add_argument("--pair_stride", type=int, default=1, help="Stride for pair comparison within window.")
    p.add_argument("--covis_thresh", type=float, default=0.5, help="Flow covisibility threshold.")
    p.add_argument("--conf_percentile", type=float, default=20.0, help="Depth confidence percentile.")
    p.add_argument("--conf_min", type=float, default=0.2, help="Depth confidence min.")
    p.add_argument("--tau_z", type=float, default=0.02, help="Occlusion depth margin.")

    # Output
    p.add_argument("--csv_dir", type=str, default=".", help="Directory for the per-model CSVs.")
    p.add_argument("--print_per_clip", action="store_true", help="Print per-clip scores to stdout.")

    add_backend_args(p)
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # 1. Resolve Models
    if len(args.models) == 1 and args.models[0].lower() == "all":
        model_keys = sorted([m for m in dic_model_fps.keys() if os.path.isdir(os.path.join(args.frames_root, m))])
    else:
        model_keys = args.models

    # 2. Resolve Suffixes
    suffixes_arg = [s.lower() for s in args.suffixes]
    suffixes = None if (len(suffixes_arg) == 1 and suffixes_arg[0] == "all") else suffixes_arg

    # 3. Build Task List
    tasks = []
    for model_key in model_keys:
        for cat in args.categories:
            cat_dir = os.path.join(args.frames_root, model_key, cat)
            chosen = pick_suffix_subdirs(cat_dir, suffixes)
            for clip_id in chosen:
                tasks.append((model_key, cat, clip_id, os.path.join(cat_dir, clip_id)))

    if not tasks:
        print("No clips found matching criteria.")
        return

    # 4. Load Models (one summary line per backend is printed)
    geometry, flow = build_backends(args)

    # 5. Run Evaluation
    per_model_category = {}
    per_model_clip_rows = {}

    pbar = tqdm(tasks, desc="Evaluating", unit="clip")
    for (model_key, cat, clip_id, video_dir) in pbar:
        pbar.set_postfix_str(f"{model_key}/{cat}/{clip_id}")

        try:
            m, d, f = evaluate_video_folder(video_dir, model_key, geometry, flow, args)
        except Exception as e:
            print(f"Error on {video_dir}: {e}", file=sys.stderr)
            m, d, f = float('nan'), float('nan'), float('nan')

        if args.print_per_clip:
            print(f"{model_key},{cat},{clip_id},{m:.4f},{d:.4f},{f:.4f}")

        per_model_category.setdefault((model_key, cat), []).append((m, d, f))
        per_model_clip_rows.setdefault(model_key, []).append((clip_id, cat, m, d, f))

    # 6. Save CSVs
    os.makedirs(args.csv_dir, exist_ok=True)
    for model_key, rows in per_model_clip_rows.items():
        rows_sorted = sorted(rows, key=lambda r: (r[1], r[0]))
        with open(os.path.join(args.csv_dir, f"{model_key}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_id", "category", "motion", "depth", "fused"])
            for r in rows_sorted:
                writer.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}"])

    # 7. Print Summary
    print("\n# SUMMARY_HEADER,model_key,category,n_clips,motion_mean,depth_mean,fused_mean")
    def nanmean_triplet(rows):
        arr = np.array(rows, dtype=float)
        if arr.size == 0:
            return (float('nan'),) * 3, 0
        n = int(np.sum(np.isfinite(arr).all(axis=1)))
        return tuple(np.nanmean(arr, axis=0)), n

    for (model_key, cat) in sorted(per_model_category.keys()):
        (m, d, f), n = nanmean_triplet(per_model_category[(model_key, cat)])
        print(f"# SUMMARY,{model_key},{cat},{n},{m:.6f},{d:.6f},{f:.6f}")

if __name__ == "__main__":
    main()
