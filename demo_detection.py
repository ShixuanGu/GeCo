#!/usr/bin/env python3
import os, sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'external', 'UFM'))
sys.path.append(os.path.join(base_path, 'external', 'vggt'))

import glob, json, csv, argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import imageio 

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from uniflowmatch.models.ufm import UniFlowMatchConfidence

# Assumes you have saved the cleaned utils as utils_clean.py
from utils import (
    load_image_ufm, predict_correspondences, rigid_flow_from_camera_motion,
    create_confidence_mask_torch, normalize_flow_to_unitless, vggt_infer,
    compute_normalized_depth_error_bidirectional, render_error_maps_to_memory
)


def safe_avg(s, c): return s / c.clamp_min(1.0)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate video consistency using VGGT (structure) and UFM (motion).")
    
    # Path Arguments
    p.add_argument("--frame_path", type=str, required=True, 
                   help="Full path to the directory containing the video frames (images).")
    p.add_argument("--outdir", type=str, default="results", 
                   help="Directory where output videos (mp4/gif) and images will be saved.")
    
    # Output Control
    p.add_argument("--fps", type=int, default=16, 
                   help="Frame rate for the output video visualization.")
    p.add_argument("--save_mode", type=str, default="mp4", choices=["gif", "mp4", "both"], 
                   help="Format for the output visualization: 'gif', 'mp4', or 'both'.")

    # Sliding Window Settings
    p.add_argument("--window_size", type=int, default=20, 
                   help="The size of the temporal window (number of frames) processed at each step.")
    p.add_argument("--window_anchor", type=str, default="center", choices=["start", "center", "end"],
                   help="Determines the position of the current frame within the sliding window:"
                        "start: Current frame is the first frame; looks ahead at future frames."
                        "center: Current frame is in the middle; looks at past and future neighbors."
                        "end: Current frame is the last frame; looks back at previous frames.")

    # Algorithm Thresholds
    p.add_argument("--tau_z", type=float, default=0.02, 
                   help="Relative depth margin threshold for occlusion detection.")
    p.add_argument("--covis_thresh", type=float, default=0.5, 
                   help="Covisibility threshold for the flow model (UFM).")
    p.add_argument("--conf_percentile", type=float, default=20, 
                   help="Percentile threshold for the depth confidence mask.")
    p.add_argument("--conf_min", type=float, default=0.2, 
                   help="Minimum absolute confidence value required for a pixel to be valid.")
    
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0]>=8) else torch.float32

    # Load Models
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    ufm = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(device=device, dtype=torch.float32).eval()
    
    img_dir = args.frame_path
    if not os.path.exists(img_dir):
        print(f"[ERROR] Path does not exist: {img_dir}")
        return

    fn = os.path.basename(os.path.normpath(img_dir))
    os.makedirs(args.outdir, exist_ok=True)

    with torch.inference_mode():
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        all_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.splitext(f.lower())[1] in exts])
        
        if not all_files:
            print(f"[ERROR] No images found in {img_dir}")
            return
        
        N = len(all_files)
        K = args.window_size
        
        print(f"[INFO] {fn}: Processing {N} frames. Window Size: {K}. Anchor: {args.window_anchor}")

        # --- PREPARE VIDEO WRITERS ---
        mp4_writers = {}
        gif_writers = {}
        
        all_output_keys = ["grid"]
        
        save_mp4 = args.save_mode in ["mp4", "both"]
        save_gif = args.save_mode in ["gif", "both"]
        
        mp4_keys = all_output_keys if save_mp4 else []
        gif_keys = all_output_keys if save_gif else []
        png_keys = all_output_keys 

        png_outdir = os.path.join(args.outdir, f"{fn}_frames")
        if len(png_keys) > 0:
            os.makedirs(png_outdir, exist_ok=True)

        for k in mp4_keys:
            out_name = f"{fn}_{k}.mp4"
            if k == "grid": out_name = f"{fn}.mp4"
            path = os.path.join(args.outdir, out_name)
            mp4_writers[k] = imageio.get_writer(
                path, fps=args.fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None
            )
        
        for k in gif_keys:
            out_name = f"{fn}_{k}.gif"
            if k == "grid": out_name = f"{fn}.gif"
            path = os.path.join(args.outdir, out_name)
            gif_writers[k] = imageio.get_writer(path, mode='I', fps=args.fps, loop=0)

        # --- Processing Loop (Frame by Frame) ---
        img_tmp = load_image_ufm(all_files[0])
        H, W = img_tmp.shape[:2]

        for global_idx in tqdm(range(N), desc=f"Frames ({fn})"):
            
            # ---------------------------------------------------------
            # 1. SLIDING WINDOW INDEXING
            # Determine the range of frames [start, end) to use as context.
            # ---------------------------------------------------------
            if args.window_anchor == "center":
                half_k = K // 2
                start_idx = global_idx - half_k
                end_idx = start_idx + K
            elif args.window_anchor == "end":
                end_idx = global_idx + 1
                start_idx = end_idx - K
            elif args.window_anchor == "start":
                start_idx = global_idx
                end_idx = start_idx + K
            else:
                # Default fallback
                start_idx = global_idx
                end_idx = start_idx + K

            # Clamp indices to ensure they stay within valid video bounds [0, N]
            if start_idx < 0:
                start_idx = 0
                end_idx = min(N, K)
            if end_idx > N:
                end_idx = N
                start_idx = max(0, N - K)

            # ---------------------------------------------------------
            # 2. BATCH PREPARATION
            # Construct a batch containing the current frame (Source) 
            # and its temporal neighbors (Targets).
            # ---------------------------------------------------------
            neighbor_indices = [i for i in range(start_idx, end_idx) if i != global_idx]
            batch_indices = [global_idx] + neighbor_indices
            batch_files = [all_files[i] for i in batch_indices]
            num_curr = len(batch_files)

            # --- RUN MODEL ---
            imgs = load_and_preprocess_images(batch_files, "crop", patch_size=14).to(device)[None]
            vggt_res = vggt_infer(vggt, imgs, (H, W), False, dtype, device)
            K_cam, E_cam, D, C = vggt_res['intrinsic'], vggt_res['extrinsic'], vggt_res['depth_map'], vggt_res['vggt_conf']
            
            src_img = load_image_ufm(batch_files[0])
            
            # Accumulators for the current frame (averaged over all neighbors)
            mot_sum, mot_cnt = torch.zeros((H,W), device=device), torch.zeros((H,W), device=device)
            dep_sum, dep_cnt = torch.zeros((H,W), device=device), torch.zeros((H,W), device=device)
            fus_sum, fus_cnt = torch.zeros((H,W), device=device), torch.zeros((H,W), device=device)

            conf_src = create_confidence_mask_torch(C[0, ..., 0], args.conf_percentile, args.conf_min)

            # Compare Src (Index 0) vs All Neighbors
            for ti in range(1, num_curr):
                tgt_img = load_image_ufm(batch_files[ti])
                flow, cov = predict_correspondences(ufm, src_img, tgt_img, "255")
                ego, mask_rp = rigid_flow_from_camera_motion(D[0], K_cam[[0, ti]], E_cam[[0, ti]])
                
                mask_rp = mask_rp & conf_src
                mask_cov = torch.isfinite(cov) & (cov > args.covis_thresh)

                res_uv = normalize_flow_to_unitless(flow - ego, K_cam[0])
                e_xy = torch.linalg.vector_norm(res_uv, dim=-1)
                
                e_z, val, _, dz, dz_p = compute_normalized_depth_error_bidirectional(
                    D[0], K_cam[0], E_cam[0], D[ti], K_cam[ti], E_cam[ti], align_corners=True
                )
                dep_valid = val & conf_src

                # ---------------------------------------------------------
                # 3. FUSED ERROR
                # Determine how to combine errors based on occlusion status.
                # ---------------------------------------------------------
                # Case 1: Consistent Motion (Flow valid + Reproj valid)
                mot_valid = mask_cov & mask_rp
                
                # Case 2: Non-Occluded (Motion valid + Depth check passes)
                dep_no_occ = mot_valid & dep_valid
                
                # Case 3: Wrong Occlusion (Flow says "occluded", but Depth says "in front")
                # This catches hallucinated occlusions or flow inconsistencies.
                wrong_occ = (~mask_cov) & mask_rp & dep_valid & ~(dep_valid & (dz_p > args.tau_z))

                # Accumulate Motion and Depth Errors
                mot_sum += e_xy * mot_valid.float(); mot_cnt += mot_valid.float()
                dep_sum += e_z * (dep_no_occ | wrong_occ).float(); dep_cnt += (dep_no_occ | wrong_occ).float()
                
                # Fused Error Map Construction:
                p_fus = torch.zeros_like(e_xy)
                
                # A. If Non-Occluded: Average of Motion & Depth error
                p_fus[dep_no_occ] = (e_xy[dep_no_occ] + e_z[dep_no_occ])/2
                
                # B. If Wrong Occlusion: Use Depth error only (Motion is unreliable)
                p_fus[wrong_occ] = e_z[wrong_occ]
                
                # C. If Motion Valid but Depth Invalid: Use Motion error only
                p_fus[mot_valid & ~dep_no_occ] = e_xy[mot_valid & ~dep_no_occ]
                
                fus_sum += p_fus; fus_cnt += (dep_no_occ | wrong_occ | (mot_valid & ~dep_no_occ)).float()

            # Average over the window
            avg_m = safe_avg(mot_sum, mot_cnt).cpu()
            avg_d = safe_avg(dep_sum, dep_cnt).cpu()
            avg_f = safe_avg(fus_sum, fus_cnt).cpu()
            
            # RENDER
            rendered_frames = render_error_maps_to_memory(
                src_img,
                motion_error_2d=avg_m, depth_error=avg_d,
                fused_error=avg_f,
                titles=(f"Src {global_idx}", "Motion Map", "Structure Map", "Fused Map")
            )
            
            # STREAM TO MP4
            for k, writer in mp4_writers.items():
                if k in rendered_frames:
                    writer.append_data(rendered_frames[k])

            # SAVE PNGs AND STREAM TO GIF
            for k in png_keys:
                if k in rendered_frames:
                    frame_data = rendered_frames[k]
                    png_name = f"{k}_{global_idx:04d}.png"
                    imageio.imwrite(os.path.join(png_outdir, png_name), frame_data)
                    if k in gif_writers:
                        gif_writers[k].append_data(frame_data)
        
        # CLOSE WRITERS
        for writer in mp4_writers.values(): writer.close()
        for writer in gif_writers.values(): writer.close()
        
        print(f"[DONE] {fn}: Saved outputs for mode={args.save_mode}.")

if __name__ == "__main__":
    main()