#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Benchmark Evaluator for Video Consistency.

This script iterates over a structured dataset of video frames (Model -> Category -> Clip),
computes consistency scores using VGGT (structure) and UFM (motion), and aggregates results.

Outputs:
1. Per-clip CSVs: Saved as `<model_key>.csv` (clip_id, category, motion, depth, fused).
2. Summary: Prints aggregated mean scores per category to stdout.

Sampling Strategy:
Instead of processing every frame (which is slow), this script divides the video into 
a fixed number of temporal windows (e.g., 4 windows of 3 seconds each) and computes 
the score within those windows.
"""

import sys
import os
import math
import argparse
import random
import csv
import numpy as np
from typing import List, Tuple

# Add paths to external submodules
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'external', 'UFM'))
sys.path.append(os.path.join(base_path, 'external', 'vggt'))

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, total=None, desc=None, unit=None): return it

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from uniflowmatch.models.ufm import UniFlowMatchConfidence

from utils import (
    load_image_ufm, predict_correspondences, rigid_flow_from_camera_motion,
    create_confidence_mask_torch, normalize_flow_to_unitless, vggt_infer,
    compute_normalized_depth_error_unidirectional,masked_mean, build_fixed_len_cover_windows
)

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

@torch.inference_mode()
def evaluate_one_window(image_files, vggt_model, ufm_model, device, compute_dtype, args):
    """
    Evaluates consistency for a single window of frames.
    
    Steps:
    1. Run VGGT once on the whole window to get Camera Poses & Depth.
    2. Iterate through pairs (Source -> Target).
    3. Compute Motion Error (Flow vs Ego-motion).
    4. Compute Depth Error (Bidirectional Reprojection).
    5. Fuse errors based on occlusion logic.
    """
    # Load first image to get dimensions
    img0 = load_image_ufm(image_files[0]) 
    H, W = img0.shape[:2]

    # Cache UFM images to reduce disk I/O latency
    cached_imgs = [img0] + [load_image_ufm(fp) for fp in image_files[1:]]

    # 1. VGGT Inference (Batch)
    imgs = load_and_preprocess_images(image_files, "crop", patch_size=14).to(device)[None]
    geo = vggt_infer(
        vggt_model, imgs, upsample_size=(H, W),
        point_prediction=False, compute_dtype=compute_dtype, device=device
    )
    K = geo["intrinsic"]   # (N,3,3)
    E = geo["extrinsic"]   # (N,3,4)
    D = geo["depth_map"]   # (N,H,W,1)
    C = geo["vggt_conf"]   # (N,H,W,1)

    N = len(image_files)
    per_frame_motion, per_frame_depth, per_frame_fused = [], [], []

    step = max(1, int(args.pair_stride))

    for src in range(N):
        src_img = cached_imgs[src]

        # Accumulators
        mot_sum, mot_cnt = torch.zeros((H, W), device=device), torch.zeros((H, W), device=device)
        dep_sum, dep_cnt = torch.zeros((H, W), device=device), torch.zeros((H, W), device=device)
        fus_sum, fus_cnt = torch.zeros((H, W), device=device), torch.zeros((H, W), device=device)

        # Source Confidence Mask
        conf_src_mask = create_confidence_mask_torch(
            C[src, ..., 0], percentile_val=args.conf_percentile, min_threshold=args.conf_min
        )

        for tgt in range(0, N, step):
            if tgt == src: continue

            tgt_img = cached_imgs[tgt]

            # 2. UFM Flow & Covisibility
            flow_uv, cov = predict_correspondences(ufm_model, src_img, tgt_img, str(args.ufm_longside))
            mask_covis = torch.isfinite(cov) & (cov > args.covis_thresh)

            # 3. Ego Motion Check
            ego_uv, mask_reproj = rigid_flow_from_camera_motion(D[src], K[[src, tgt]], E[[src, tgt]])
            mask_reproj = mask_reproj & conf_src_mask

            # Motion Residual
            residual_uv = flow_uv - ego_uv
            residual_unitless = normalize_flow_to_unitless(residual_uv, K[src])
            e_xy = torch.linalg.vector_norm(residual_unitless, dim=-1)

            # 4. Depth Consistency (Unidirectional for efficient computing)
            # Use the robust check to avoid false positives in occluded regions
            e_z, dp_valid_basic, _, dz, dz_rel_p = compute_normalized_depth_error_unidirectional(
                D[src], K[src], E[src], D[tgt], K[tgt], E[tgt], align_corners=True
            )
            depth_valid = dp_valid_basic & conf_src_mask

            # 5. Fusion Logic
            mot_valid      = mask_covis & mask_reproj
            depth_no_occ   = mask_covis & mask_reproj & depth_valid
            wrong_occ      = (~mask_covis) & mask_reproj & depth_valid & ~(depth_valid & (dz_rel_p > args.tau_z))

            # Accumulate
            mot_sum += e_xy * mot_valid.float()
            mot_cnt += mot_valid.float()

            depth_mask = depth_no_occ | wrong_occ
            dep_sum   += e_z * depth_mask.float()
            dep_cnt   += depth_mask.float()

            pair_fused = torch.zeros_like(e_xy)
            pair_fused[depth_no_occ] = (e_xy[depth_no_occ] + e_z[depth_no_occ]) / 2.0
            pair_fused[wrong_occ] = e_z[wrong_occ]
            pair_fused[mot_valid & ~depth_no_occ] = e_xy[mot_valid & ~depth_no_occ] # Motion only fallback

            fus_sum += pair_fused
            fus_cnt += (depth_no_occ | wrong_occ | (mot_valid & ~depth_no_occ)).float()

        # Compute scalars for this frame
        def safe_div(s, c): return s / c.clamp_min(1.0)
        
        motion_avg = safe_div(mot_sum, mot_cnt)
        depth_avg  = safe_div(dep_sum, dep_cnt)
        fused_avg  = safe_div(fus_sum, fus_cnt)

        per_frame_motion.append(masked_mean(motion_avg, mot_cnt > 0))
        per_frame_depth.append(masked_mean(depth_avg, dep_cnt > 0))
        per_frame_fused.append(masked_mean(fused_avg, fus_cnt > 0))

    # Window Averaging
    def nanmean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)) if arr.size > 0 else float('nan')

    return nanmean(per_frame_motion), nanmean(per_frame_depth), nanmean(per_frame_fused), len(image_files)


def evaluate_video_folder(video_dir, model_key, vggt_model, ufm_model, device, compute_dtype, args):
    """Entry point for a single video folder."""
    def list_frame_files(d):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        return sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f.lower())[1] in exts])

    all_files = list_frame_files(video_dir)
    if len(all_files) < 2:
        return float('nan'), float('nan'), float('nan')

    # Get Sampling Windows
    fps_native = float(dic_model_fps[model_key])
    eval_fps_eff = min(float(args.eval_fps), fps_native)

    windows, _ = build_fixed_len_cover_windows(
        all_files=all_files,
        fps_native=fps_native,
        eval_fps=eval_fps_eff,
        win_sec=args.win_sec,
        max_windows=args.max_windows
    )
    
    if not windows: return float('nan'), float('nan'), float('nan')

    # Run windows
    motion_vals, depth_vals, fused_vals, weights = [], [], [], []
    for wfiles in windows:
        m, d, f, w = evaluate_one_window(wfiles, vggt_model, ufm_model, device, compute_dtype, args)
        if np.isfinite(m) and np.isfinite(d) and np.isfinite(f) and w >= 2:
            motion_vals.append(m); depth_vals.append(d); fused_vals.append(f); weights.append(w)

    if not weights: return float('nan'), float('nan'), float('nan')

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
    if not os.path.isdir(cat_dir): return []
    subs = [d for d in sorted(os.listdir(cat_dir)) if os.path.isdir(os.path.join(cat_dir, d))]
    
    if suffixes is None: return subs
    
    keep = []
    suffixes = [str(s) for s in suffixes]
    for d in subs:
        for s in suffixes:
            if d.endswith(f"_{s}"):
                keep.append(d)
                break
    return keep

# ---------------------- Main Execution ----------------------
def parse_args():
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

    # Algorithm
    p.add_argument("--pair_stride", type=int, default=1, help="Stride for pair comparison within window.")
    p.add_argument("--covis_thresh", type=float, default=0.5, help="Flow covisibility threshold.")
    p.add_argument("--conf_percentile", type=float, default=20.0, help="Depth confidence percentile.")
    p.add_argument("--conf_min", type=float, default=0.2, help="Depth confidence min.")
    p.add_argument("--tau_z", type=float, default=0.02, help="Occlusion depth margin.")
    p.add_argument("--ufm_longside", type=int, default=255, help="UFM inference resolution.")

    p.add_argument("--print_per_clip", action="store_true", help="Print per-clip scores to stdout.")
    return p.parse_args()

def main():
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

    # 4. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    ufm_model  = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(dtype=torch.float32, device=device).eval()

    # 5. Run Evaluation
    per_model_category = {}
    per_model_clip_rows = {}

    pbar = tqdm(tasks, desc="Evaluating", unit="clip")
    for (model_key, cat, clip_id, video_dir) in pbar:
        pbar.set_postfix_str(f"{model_key}/{cat}/{clip_id}")
        
        try:
            m, d, f = evaluate_video_folder(
                video_dir, model_key, vggt_model, ufm_model, device, compute_dtype, args
            )
        except Exception as e:
            print(f"Error on {video_dir}: {e}", file=sys.stderr)
            m, d, f = float('nan'), float('nan'), float('nan')

        if args.print_per_clip:
            print(f"{model_key},{cat},{clip_id},{m:.4f},{d:.4f},{f:.4f}")

        per_model_category.setdefault((model_key, cat), []).append((m, d, f))
        per_model_clip_rows.setdefault(model_key, []).append((clip_id, cat, m, d, f))

    # 6. Save CSVs
    for model_key, rows in per_model_clip_rows.items():
        rows_sorted = sorted(rows, key=lambda r: (r[1], r[0]))
        with open(f"{model_key}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_id", "category", "motion", "depth", "fused"])
            for r in rows_sorted:
                writer.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}"])

    # 7. Print Summary
    print("\n# SUMMARY_HEADER,model_key,category,n_clips,motion_mean,depth_mean,fused_mean")
    def nanmean_triplet(rows):
        arr = np.array(rows, dtype=float)
        if arr.size == 0: return (float('nan'),)*3, 0
        n = int(np.sum(np.isfinite(arr).all(axis=1)))
        return tuple(np.nanmean(arr, axis=0)), n

    for (model_key, cat) in sorted(per_model_category.keys()):
        (m, d, f), n = nanmean_triplet(per_model_category[(model_key, cat)])
        print(f"# SUMMARY,{model_key},{cat},{n},{m:.6f},{d:.6f},{f:.6f}")

if __name__ == "__main__":
    main()