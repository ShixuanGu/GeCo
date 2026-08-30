#!/usr/bin/env python3
"""
Per-video GeCo consistency maps.

For every frame of a video the script builds a temporal window around it, runs
the geometry backend (camera + depth) once on the window, predicts the flow from
the frame to each neighbour with the flow backend, and accumulates the GeCo
motion / structure / fused errors (``geco.metric``).  The averaged maps are
rendered next to the source frame and streamed to an mp4 / gif plus per-frame
PNGs.

Backends are selected with the shared ``--geometry`` / ``--flow`` flags
(``geco.cli``); the default pair (VGGT + UFM) is the one used in the paper.
"""
from __future__ import annotations

import argparse
import os

import imageio
import torch
from tqdm import tqdm

import geco  # noqa: F401  (puts the vendored third-party packages on sys.path)
from geco.cli import add_backend_args, build_backends
from geco.image_io import check_same_size, list_frame_files, load_image_rgb
from geco.metric import ErrorAccumulator, compute_pair_errors, source_confidence_mask
from utils import render_error_maps_to_memory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate video consistency with a geometry backend (structure) and a flow backend (motion).")

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
                   help="Covisibility threshold for the flow model.")
    p.add_argument("--conf_percentile", type=float, default=20,
                   help="Percentile threshold for the depth confidence mask.")
    p.add_argument("--conf_min", type=float, default=0.2,
                   help="Minimum absolute confidence value required for a pixel to be valid.")

    add_backend_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    img_dir = args.frame_path
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Path does not exist: {img_dir}")
        return

    # Load Models (one summary line per backend is printed)
    geometry, flow = build_backends(args)
    device = geometry.device

    fn = os.path.basename(os.path.normpath(img_dir))
    os.makedirs(args.outdir, exist_ok=True)

    with torch.no_grad():
        all_files = list_frame_files(img_dir)

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
            if k == "grid":
                out_name = f"{fn}.mp4"
            path = os.path.join(args.outdir, out_name)
            mp4_writers[k] = imageio.get_writer(
                path, fps=args.fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None
            )

        for k in gif_keys:
            out_name = f"{fn}_{k}.gif"
            if k == "grid":
                out_name = f"{fn}.gif"
            path = os.path.join(args.outdir, out_name)
            gif_writers[k] = imageio.get_writer(path, mode='I', fps=args.fps, loop=0)

        # --- Processing Loop (Frame by Frame) ---
        H, W = check_same_size(all_files)

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

            # --- RUN GEOMETRY (cameras + depth + confidence on the frames' native grid) ---
            geo = geometry.run(batch_files, output_hw=(H, W))

            src_img = load_image_rgb(batch_files[0])

            # Accumulators for the current frame (averaged over all neighbors)
            acc = ErrorAccumulator((H, W), device)

            conf_src = source_confidence_mask(geo, 0, args.conf_percentile, args.conf_min)

            # Compare Src (Index 0) vs All Neighbors
            for ti in range(1, num_curr):
                tgt_img = load_image_rgb(batch_files[ti])
                flow_out = flow.predict(src_img, tgt_img)

                # ---------------------------------------------------------
                # 3. MOTION / STRUCTURE / FUSED ERROR
                # Flow vs. ego-motion, bidirectional depth re-projection and
                # the occlusion-aware fusion (see geco.metric).
                # ---------------------------------------------------------
                pe = compute_pair_errors(
                    geo, 0, ti, flow_out, conf_src,
                    covis_thresh=args.covis_thresh, tau_z=args.tau_z, depth_check="bidirectional",
                )
                acc.add(pe)

            # Average over the window
            maps = acc.maps()
            avg_m = maps["motion"].cpu()
            avg_d = maps["depth"].cpu()
            avg_f = maps["fused"].cpu()

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
        for writer in mp4_writers.values():
            writer.close()
        for writer in gif_writers.values():
            writer.close()

        print(f"[DONE] {fn}: Saved outputs for mode={args.save_mode}.")


if __name__ == "__main__":
    main()
