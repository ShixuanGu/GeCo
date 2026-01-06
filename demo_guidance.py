#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
external_root = os.path.join(current_dir, "external")
paths_to_add = [
    os.path.join(external_root, "guidance_cogvideox"),
    os.path.join(external_root, "vggt"),
    os.path.join(external_root, "UFM"),
    external_root 
]
for path in paths_to_add:
    if os.path.exists(path):
        if path not in sys.path:
            sys.path.append(path)
    else:
        print(f"Warning: Dependency path not found: {path}")

try:
    from cogvideox import CogVideoXPipeline
except ImportError as e:
    print(f"Error: Could not import modules. Ensure the 'external' folder is present.\nDetails: {e}")
    sys.exit(1)

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

from diffusers.utils import export_to_gif, export_to_video
from cogvideox import CogVideoXPipeline

from utils_guidance import (
    vggt_infer,
    rigid_flow_from_camera_motion,
    normalize_flow_to_unitless,
    _ufm_flow_with_grad,
    create_confidence_mask_torch
)

# ---------------- helpers ----------------

def parse_line_to_list(line: str) -> List[int]:
    out = []
    for part in line.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out

def auto_memory_profile(pipe, profile: str):
    profile = profile.lower()
    total_mem_gb = None
    try:
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / (1024**3)
    except Exception:
        pass
    if profile == "auto" and total_mem_gb is not None:
        if total_mem_gb >= 80:
            profile = "high"
        elif total_mem_gb >= 40:
            profile = "medium"
        else:
            profile = "low"
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
    if profile == "high":
        if hasattr(pipe.vae, "disable_tiling"): pipe.vae.disable_tiling()
        if hasattr(pipe.vae, "disable_slicing"): pipe.vae.disable_slicing()
        if hasattr(pipe.vae, "disable_gradient_checkpointing"): pipe.vae.disable_gradient_checkpointing()
    elif profile == "medium":
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_gradient_checkpointing"): pipe.vae.enable_gradient_checkpointing()
    else:
        if hasattr(pipe.vae, "enable_tiling"): pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_gradient_checkpointing"): pipe.vae.enable_gradient_checkpointing()

def build_pipeline(model_path: str, dtype: str, memory_profile: str) -> CogVideoXPipeline:
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(dtype.lower(), torch.float16)
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch_dtype).to("cuda")
    auto_memory_profile(pipe, memory_profile)
    return pipe

def _frames01_to_vggt_crop(frames_01: torch.Tensor, target_width: int = 518, patch_size: int = 14) -> torch.Tensor:
    x = frames_01[0].to(dtype=torch.float32)   # [K,H,W,3]
    x = x.permute(0, 3, 1, 2).contiguous()     # [K,3,H,W]
    _, _, H0, W0 = x.shape
    scale = target_width / float(W0)
    H_scaled = int(round((H0 * scale) / patch_size) * patch_size)
    H_scaled = max(patch_size, H_scaled)
    x = F.interpolate(x, size=(H_scaled, target_width), mode="bilinear", align_corners=False)
    if H_scaled > target_width:
        top = (H_scaled - target_width) // 2
        x = x[:, :, top:top + target_width, :]
    return x.unsqueeze(0)  # [1,K,3,H',518]

def _percentile_conf_mask(conf_hw1: torch.Tensor, percentile: float, min_threshold: float) -> torch.Tensor:
    c = conf_hw1[..., 0]
    finite = torch.isfinite(c)
    if finite.any():
        q = torch.quantile(c[finite], percentile / 100.0)
        thr = torch.maximum(q, torch.tensor(min_threshold, device=c.device, dtype=c.dtype))
    else:
        thr = torch.tensor(min_threshold, device=c.device, dtype=c.dtype)
    return finite & (c >= thr)

def _resize_flow_to(flow_hw2: torch.Tensor, H: int, W: int, srcH: int, srcW: int) -> torch.Tensor:
    if flow_hw2.shape[0] == H and flow_hw2.shape[1] == W:
        return flow_hw2
    flow = flow_hw2.permute(2,0,1).unsqueeze(0)
    flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
    flow[:,0] *= (W / float(srcW))
    flow[:,1] *= (H / float(srcH))
    return flow[0].permute(1,2,0)

def _ufm_flow_scaled(ufm_model, src_hw3: torch.Tensor, tgt_hw3: torch.Tensor, scale: float):
    H, W = src_hw3.shape[:2]
    if scale < 1.0:
        Hs = max(1, int(round(H * scale)))
        Ws = max(1, int(round(W * scale)))
        src_small = F.interpolate(src_hw3.permute(2,0,1).unsqueeze(0), size=(Hs, Ws), mode='bilinear', align_corners=False)[0].permute(1,2,0)
        tgt_small = F.interpolate(tgt_hw3.permute(2,0,1).unsqueeze(0), size=(Hs, Ws), mode='bilinear', align_corners=False)[0].permute(1,2,0)
        flow_s, cov_s = _ufm_flow_with_grad(ufm_model, src_small, tgt_small)
        flow = _resize_flow_to(flow_s, H, W, Hs, Ws)
        cov  = F.interpolate(cov_s.unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False)[0,0]
        return flow, cov
    else:
        return _ufm_flow_with_grad(ufm_model, src_hw3, tgt_hw3)

def _build_pairs(K: int, mode: str):
    if K < 2:
        return []
    if mode == "adjacent":
        return [(i, i+1) for i in range(K-1)]
    pairs = []
    for s in range(K):
        for t in range(K):
            if s != t:
                pairs.append((s,t))
    return pairs

def _get_compute_dtype_for_vggt(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

# depth reprojection metric
def compute_normalized_depth_error(
    depth_src_hw1: torch.Tensor, K_src_33: torch.Tensor, E_src_34: torch.Tensor,
    depth_tgt_hw1: torch.Tensor, K_tgt_33: torch.Tensor, E_tgt_34: torch.Tensor,
    *, align_corners=True
):
    device = depth_src_hw1.device
    H, W = depth_src_hw1.shape[:2]
    Zs = depth_src_hw1[..., 0].float().clamp(min=1e-6)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3).T  # (3, N)

    Kinv = torch.inverse(K_src_33.float())
    Xs = (Kinv @ pix) * Zs.reshape(1, -1)  # (3, N)

    Rs = E_src_34[:3, :3].float()
    ts = E_src_34[:3, 3].float()
    Rt = E_tgt_34[:3, :3].float()
    tt = E_tgt_34[:3, 3].float()

    R_ts = Rt @ Rs.T
    t_ts = tt - Rt @ Rs.T @ ts
    Xt = (R_ts @ Xs) + t_ts[:, None]       # (3, N)
    Zt = Xt[2, :].clamp(min=1e-6)

    fx_t, fy_t = K_tgt_33[0, 0].float(), K_tgt_33[1, 1].float()
    cx_t, cy_t = K_tgt_33[0, 2].float(), K_tgt_33[1, 2].float()
    u = fx_t * (Xt[0, :] / Zt) + cx_t
    v = fy_t * (Xt[1, :] / Zt) + cy_t

    Ht, Wt = depth_tgt_hw1.shape[:2]
    inside = (u >= 0) & (u <= (Wt - 1)) & (v >= 0) & (v <= (Ht - 1))
    finite = torch.isfinite(Zs.reshape(-1)) & torch.isfinite(Zt)
    valid = (inside & finite).reshape(H, W)

    u_norm = (u / (Wt - 1)) * 2 - 1
    v_norm = (v / (Ht - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).reshape(H, W, 2)[None, ...]

    D_tgt = depth_tgt_hw1.permute(2, 0, 1).float().unsqueeze(0)  # 1,1,Ht,Wt
    D_tgt_smpl = F.grid_sample(D_tgt, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)[0, 0]  # (H,W)

    valid = valid & torch.isfinite(D_tgt_smpl) & (D_tgt_smpl > 0)

    Zpred = Zt.reshape(H, W)
    deltaZ = D_tgt_smpl - Zpred
    e_z = deltaZ.abs() / Zs

    dz_rel = (Zpred - D_tgt_smpl) / Zs
    dz_rel_p = torch.clamp(dz_rel, min=0)
    return e_z, valid, Zs, dz_rel, dz_rel_p

# metric builders
def make_motion_metric(
    vggt_model, ufm_model, device, compute_dtype,
    debug_once_dir=None,
    vggt_strategy: str = "once",
    pair_mode: str = "adjacent",
    ufm_scale: float = 0.5,
    cov_thresh: float = 0.5,
    percentile_val: int = 30,
    min_threshold: float = 0.3,
    grad_through_vggt: bool = False,
    debug_autograd: bool = False,
):
    import torch
    import torch.nn.functional as F

    assert vggt_strategy in ("once", "per_source")
    assert pair_mode in ("adjacent", "all")
    assert 0.0 < ufm_scale <= 1.0
    printed = False

    def _frames01_to_vggt_crop(frames_01: torch.Tensor, target_width: int = 518, patch_size: int = 14) -> torch.Tensor:
        x = frames_01[0].to(dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).contiguous()
        _, _, H0, W0 = x.shape
        scale = target_width / float(W0)
        H_scaled = int(round((H0 * scale) / patch_size) * patch_size)
        H_scaled = max(patch_size, H_scaled)
        x = F.interpolate(x, size=(H_scaled, target_width), mode="bilinear", align_corners=False)
        if H_scaled > target_width:
            top = (H_scaled - target_width) // 2
            x = x[:, :, top:top + target_width, :]
        return x.unsqueeze(0)

    def _resize_HW1(x_hw1: torch.Tensor, H2: int, W2: int) -> torch.Tensor:
        x4d = x_hw1.permute(2, 0, 1).unsqueeze(0)
        x4d2 = F.interpolate(x4d, size=(H2, W2), mode="bilinear", align_corners=False)
        return x4d2.squeeze(0).permute(1, 2, 0)

    def _scale_intrinsics(K: torch.Tensor, s: float) -> torch.Tensor:
        S = torch.tensor([[s, 0, 0], [0, s, 0], [0, 0, 1]], device=K.device, dtype=K.dtype)
        return S @ K

    def _run_vggt(frames_01: torch.Tensor):
        upsample_size = frames_01[0, 0].shape[:2]
        imgs = _frames01_to_vggt_crop(frames_01)
        out = vggt_infer(
            vggt_model, imgs,
            upsample_size=upsample_size,
            point_prediction=False,
            compute_dtype=compute_dtype,
            device=device,
            enable_grad=grad_through_vggt,  # single source of truth
        )
        return out


    def _pair_residual_sum_and_count(
        src_img_HW3: torch.Tensor,
        tgt_img_HW3: torch.Tensor,
        depth_src_HW1: torch.Tensor,
        K_pair_2x33: torch.Tensor,
        E_pair_2x34: torch.Tensor,
        conf_src_HW1: torch.Tensor,
    ):
        H, W = src_img_HW3.shape[:2]
        s = ufm_scale
        if s != 1.0:
            Hs, Ws = max(1, int(round(H * s))), max(1, int(round(W * s)))
            src_s = F.interpolate(src_img_HW3.permute(2, 0, 1).unsqueeze(0), size=(Hs, Ws),
                                  mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).contiguous()
            tgt_s = F.interpolate(tgt_img_HW3.permute(2, 0, 1).unsqueeze(0), size=(Hs, Ws),
                                  mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).contiguous()
            depth_s = _resize_HW1(depth_src_HW1, Hs, Ws)
            conf_s  = _resize_HW1(conf_src_HW1,  Hs, Ws)
            K_pair_s = torch.stack([_scale_intrinsics(K_pair_2x33[0], s), _scale_intrinsics(K_pair_2x33[1], s)], dim=0)

            # ego-flow (now possibly with grad wrt depth/K/E if grad_through_vggt=True)
            ego_flow_s, mask_reproj_s = rigid_flow_from_camera_motion(depth_s, K_pair_s, E_pair_2x34, target_size=(Hs, Ws))
            flow_s, cov_s = _ufm_flow_with_grad(ufm_model, src_s.float(), tgt_s.float())
            mask_covis_s = torch.isfinite(cov_s) & (cov_s > cov_thresh)

            conf_mask_s = create_confidence_mask_torch(conf_s[..., 0], percentile_val=percentile_val, min_threshold=min_threshold)
            valid_s = mask_reproj_s & mask_covis_s & conf_mask_s

            residual_s = (flow_s - ego_flow_s)
            rmag_s = torch.linalg.vector_norm(residual_s, dim=-1)
            n = valid_s.sum()
            sum_res = rmag_s[valid_s].sum() if n.item() > 0 else rmag_s.mean()
            cnt     = n if n.item() > 0 else torch.tensor(1, device=rmag_s.device, dtype=torch.long)
            return sum_res, cnt

        flow, cov = _ufm_flow_with_grad(ufm_model, src_img_HW3.float(), tgt_img_HW3.float())
        mask_covis = torch.isfinite(cov) & (cov > cov_thresh)
        ego_flow, mask_reproj = rigid_flow_from_camera_motion(depth_src_HW1, K_pair_2x33, E_pair_2x34)
        conf_mask = create_confidence_mask_torch(conf_src_HW1[..., 0], percentile_val=percentile_val, min_threshold=min_threshold)
        valid = mask_reproj & mask_covis & conf_mask

        residual = (flow - ego_flow)
        rmag = torch.linalg.vector_norm(residual, dim=-1)
        n = valid.sum()
        sum_res = rmag[valid].sum() if n.item() > 0 else rmag.mean()
        cnt     = n if n.item() > 0 else torch.tensor(1, device=rmag.device, dtype=torch.long)
        return sum_res, cnt

    def my_metric(frames_01: torch.Tensor) -> torch.Tensor:
        nonlocal printed
        B, F, H, W, _ = frames_01.shape
        assert B == 1, "metric assumes batch size 1"
        geo = _run_vggt(frames_01)
        K_all, E_all, D_all, C_all = geo["intrinsic"], geo["extrinsic"], geo["depth_map"], geo["vggt_conf"]
        total_sum = torch.zeros((), device=frames_01.device, dtype=torch.float32)
        total_cnt = torch.zeros((), device=frames_01.device, dtype=torch.long)
        pairs = [(i, i + 1) for i in range(F - 1)]
        for (src_idx, tgt_idx) in pairs:
            src_img = frames_01[0, src_idx]
            tgt_img = frames_01[0, tgt_idx]
            depth_s = D_all[src_idx]
            conf_s  = C_all[src_idx]
            K_pair  = K_all[[src_idx, tgt_idx]]
            E_pair  = E_all[[src_idx, tgt_idx]]
            s, n = _pair_residual_sum_and_count(src_img, tgt_img, depth_s, K_pair, E_pair, conf_s)
            total_sum = total_sum + s
            total_cnt = total_cnt + n
        mean_residual = total_sum / total_cnt.clamp_min(1)
        score = -mean_residual
        if debug_autograd and not printed:
            print(f"[DBG] frames_01.requires_grad={frames_01.requires_grad}")
            print(f"[DBG] score.requires_grad={score.requires_grad}")
            printed = True
        return score

    return my_metric


def make_depth_metric(
    vggt_model, ufm_model, device, compute_dtype,
    *, pair_mode: str, ufm_scale: float,
    cov_thresh: float, conf_percentile: float, conf_min: float, tau_z: float,
    grad_through_vggt: bool
):
    def metric(frames_01: torch.Tensor) -> torch.Tensor:
        assert frames_01.ndim == 5 and frames_01.shape[0] == 1
        _, K, H, W, _ = frames_01.shape
        imgs_cropped = _frames01_to_vggt_crop(frames_01)
        geo = vggt_infer(
            vggt_model, imgs_cropped, upsample_size=(H, W),
            point_prediction=False, compute_dtype=compute_dtype, device=device,
            enable_grad=grad_through_vggt
        )
        D_all = geo["depth_map"]
        K_all = geo["intrinsic"]
        E_all = geo["extrinsic"]
        C_all = geo.get("vggt_conf", None)

        pairs = _build_pairs(K, pair_mode)
        depth_sum = torch.zeros([], device=device, dtype=frames_01.dtype)
        depth_cnt = torch.zeros([], device=device, dtype=frames_01.dtype)

        for (s, t) in pairs:
            src = frames_01[0, s]
            tgt = frames_01[0, t]

            flow_uv, cov = _ufm_flow_scaled(ufm_model, src, tgt, ufm_scale)
            mask_covis = torch.isfinite(cov) & (cov > cov_thresh)

            K_pair = torch.stack([K_all[s], K_all[t]], dim=0)
            E_pair = torch.stack([E_all[s], E_all[t]], dim=0)
            _, mask_reproj = rigid_flow_from_camera_motion(D_all[s], K_pair, E_pair, target_size=(H, W))

            if C_all is not None:
                conf_src_mask = _percentile_conf_mask(C_all[s], conf_percentile, conf_min)
            else:
                conf_src_mask = torch.ones((H, W), dtype=torch.bool, device=device)

            e_z, dp_valid_basic, _, _, dz_rel_p = compute_normalized_depth_error(
                D_all[s], K_all[s], E_all[s], D_all[t], K_all[t], E_all[t]
            )
            depth_valid = dp_valid_basic & conf_src_mask

            depth_nonocc_valid = mask_covis & mask_reproj & depth_valid
            flow_occ_valid = (~mask_covis) & mask_reproj & depth_valid
            geom_occ       = depth_valid & (dz_rel_p > tau_z)
            wrong_occ      = flow_occ_valid & (~geom_occ)

            depth_mask = depth_nonocc_valid | wrong_occ
            depth_sum = depth_sum + (e_z * depth_mask.float()).sum()
            depth_cnt = depth_cnt + depth_mask.float().sum()

        eps = 1e-8
        depth_loss = depth_sum / depth_cnt.clamp_min(eps)
        return -depth_loss
    return metric

def make_fused_metric(
    vggt_model, ufm_model, device, compute_dtype,
    *, pair_mode: str, ufm_scale: float,
    cov_thresh: float, conf_percentile: float, conf_min: float, tau_z: float,
    grad_through_vggt: bool
):
    def metric(frames_01: torch.Tensor) -> torch.Tensor:
        assert frames_01.ndim == 5 and frames_01.shape[0] == 1
        _, K, H, W, _ = frames_01.shape

        imgs_cropped = _frames01_to_vggt_crop(frames_01)
        geo = vggt_infer(
            vggt_model, imgs_cropped, upsample_size=(H, W),
            point_prediction=False, compute_dtype=compute_dtype, device=device,
            enable_grad=grad_through_vggt
        )
        D_all = geo["depth_map"]
        K_all = geo["intrinsic"]
        E_all = geo["extrinsic"]
        C_all = geo.get("vggt_conf", None)

        pairs = _build_pairs(K, pair_mode)
        fused_sum = torch.zeros([], device=device, dtype=frames_01.dtype)
        fused_cnt = torch.zeros([], device=device, dtype=frames_01.dtype)

        for (s, t) in pairs:
            src = frames_01[0, s]
            tgt = frames_01[0, t]

            flow_uv, cov = _ufm_flow_scaled(ufm_model, src, tgt, ufm_scale)
            mask_covis = torch.isfinite(cov) & (cov > cov_thresh)

            K_pair = torch.stack([K_all[s], K_all[t]], dim=0)
            E_pair = torch.stack([E_all[s], E_all[t]], dim=0)
            ego_uv, mask_reproj = rigid_flow_from_camera_motion(D_all[s], K_pair, E_pair, target_size=(H, W))

            if C_all is not None:
                conf_src_mask = _percentile_conf_mask(C_all[s], conf_percentile, conf_min)
            else:
                conf_src_mask = torch.ones((H, W), dtype=torch.bool, device=device)

            residual_uv = flow_uv - ego_uv
            residual_unitless = normalize_flow_to_unitless(residual_uv, K_all[s])
            e_xy = torch.linalg.vector_norm(residual_unitless, dim=-1)

            e_z, dp_valid_basic, _, _, dz_rel_p = compute_normalized_depth_error(
                D_all[s], K_all[s], E_all[s], D_all[t], K_all[t], E_all[t]
            )
            depth_valid = dp_valid_basic & conf_src_mask

            depth_nonocc_valid = mask_covis & mask_reproj & depth_valid
            flow_occ_valid     = (~mask_covis) & mask_reproj & depth_valid
            geom_occ           = depth_valid & (dz_rel_p > tau_z)
            wrong_occ          = flow_occ_valid & (~geom_occ)

            pair_fused = torch.zeros_like(e_xy)
            cov_inds = depth_nonocc_valid
            pair_fused[cov_inds] = torch.sqrt(e_xy[cov_inds] * e_xy[cov_inds] + e_z[cov_inds] * e_z[cov_inds])
            pair_fused[wrong_occ] = e_z[wrong_occ]

            fused_mask = depth_nonocc_valid | wrong_occ
            fused_sum = fused_sum + (pair_fused * fused_mask.float()).sum()
            fused_cnt = fused_cnt + fused_mask.float().sum()

        eps = 1e-8
        fused_loss = fused_sum / fused_cnt.clamp_min(eps)
        return -fused_loss
    return metric

# ---------------- core run ----------------

def run_pass(pipe: CogVideoXPipeline,
             out_prefix: str,
             seed: int,
             num_inference_steps: int,
             num_frames: int,
             guidance_scale: float,
             guidance_step: List[int],
             guidance_lr: List[float],
             travel_time: Optional[Tuple[int, int]],
             output_dir: str,
             loss_fn: str,
             fixed_frames: List[int],
             prompt: str,
             residual_motion_metric: Optional[callable] = None,
             depth_metric_fn: Optional[callable] = None,
             fused_metric_fn: Optional[callable] = None):
    
    generator = torch.Generator(device="cuda").manual_seed(seed)

    additional_inputs = {}
    if loss_fn == "residual_motion":
        assert callable(residual_motion_metric), "residual_motion_metric must be callable(frames_01)->scalar"
        additional_inputs["residual_motion_metric"] = residual_motion_metric
    elif loss_fn == "metric_depth":
        assert callable(depth_metric_fn), "depth_metric_fn must be callable(frames_01)->scalar"
        additional_inputs["depth_metric"] = depth_metric_fn
    elif loss_fn == "metric_fused":
        assert callable(fused_metric_fn), "fused_metric_fn must be callable(frames_01)->scalar"
        additional_inputs["fused_metric"] = fused_metric_fn

    result = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=generator,
        fixed_frames=fixed_frames,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        loss_fn=loss_fn,
        additional_inputs=additional_inputs if additional_inputs else None,
        travel_time=travel_time,
    ).frames[0]

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"{out_prefix}.gif")
    mp4_path = os.path.join(output_dir, f"{out_prefix}.mp4")
    export_to_gif(result, gif_path, fps=8)
    export_to_video(result, mp4_path, fps=8)

    return result, gif_path, mp4_path

def hard_cuda_cleanup(*vars_to_del):
    for v in vars_to_del:
        try:
            del v
        except Exception:
            pass
    import gc
    gc.collect()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

def run(model_path: str,
        output_dir: str,
        dtype: str,
        memory_profile: str,
        seed: int,
        only: str,
        loss_fn: str,
        prompt: str,
        fixed_frames_line: str,
        metric_grad_through_vggt: bool,
        metric_pair_mode: str,
        metric_ufm_scale: float,
        metric_cov_thresh: float,
        metric_conf_percent: float,
        metric_conf_min: float,
        metric_tau_z: float):

    os.makedirs(output_dir, exist_ok=True)

    num_inference_steps = 50
    num_frames = 49
    guidance_scale = 6.0

    baseline_guidance_step = [0] * num_inference_steps
    baseline_guidance_lr = [3.0] * num_inference_steps
    baseline_travel_time = (5, 20)

    guided_guidance_step = [0] * 3 + [3] * 17 + [2] * 30
    guided_guidance_lr = [3.0] * num_inference_steps
    guided_travel_time = (15, 20)

    fixed_frames = parse_line_to_list(fixed_frames_line)
    print(f"[INFO] fixed_frames (1-based): {fixed_frames}")

    # Prepare metric callables
    # Since we are restricted to metric loss functions, we always load the models
    depth_metric_fn = fused_metric_fn = residual_motion_metric = None
    
    from vggt.models.vggt import VGGT
    from uniflowmatch.models.ufm import UniFlowMatchConfidence
    device = torch.device("cuda")
    compute_dtype = _get_compute_dtype_for_vggt(device)
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    ufm_model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(dtype=torch.float32, device=device).eval()
    for p in vggt_model.parameters(): p.requires_grad_(False)
    for p in ufm_model.parameters(): p.requires_grad_(False)

    if loss_fn == "residual_motion":
        residual_motion_metric = make_motion_metric(
            vggt_model=vggt_model,
            ufm_model=ufm_model,
            device=device,
            compute_dtype=compute_dtype,
            vggt_strategy="once",
            pair_mode=metric_pair_mode,
            ufm_scale=metric_ufm_scale,
            cov_thresh=metric_cov_thresh,
            percentile_val=metric_conf_percent,
            min_threshold=metric_conf_min,
            grad_through_vggt=metric_grad_through_vggt,
            debug_autograd=False,
        )
    elif loss_fn == "metric_depth":
        depth_metric_fn = make_depth_metric(
            vggt_model, ufm_model, device, compute_dtype,
            pair_mode=metric_pair_mode, ufm_scale=metric_ufm_scale,
            cov_thresh=metric_cov_thresh, conf_percentile=metric_conf_percent, conf_min=metric_conf_min, tau_z=metric_tau_z,
            grad_through_vggt=metric_grad_through_vggt,
        )
    elif loss_fn == "metric_fused":
        fused_metric_fn = make_fused_metric(
            vggt_model, ufm_model, device, compute_dtype,
            pair_mode=metric_pair_mode, ufm_scale=metric_ufm_scale,
            cov_thresh=metric_cov_thresh, conf_percentile=metric_conf_percent, conf_min=metric_conf_min, tau_z=metric_tau_z,
            grad_through_vggt=metric_grad_through_vggt,
        )

    outputs = []

    # BASELINE
    if only in ("both", "baseline"):
        pipe = build_pipeline(model_path, dtype, memory_profile)
        print("[A] Generating baseline...")
        with torch.inference_mode():
            _, gif_a, mp4_a = run_pass(
                pipe,
                out_prefix="metric_base",
                seed=seed,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                guidance_step=baseline_guidance_step,
                guidance_lr=baseline_guidance_lr,
                travel_time=baseline_travel_time,
                output_dir=output_dir,
                loss_fn=loss_fn,
                fixed_frames=fixed_frames,
                prompt=prompt,
                residual_motion_metric=residual_motion_metric,
                depth_metric_fn=depth_metric_fn,
                fused_metric_fn=fused_metric_fn,
            )
        outputs.extend([gif_a, mp4_a])
        print(f"[A] Saved: {gif_a}\n[A] Saved: {mp4_a}")
        hard_cuda_cleanup(pipe)

    # GUIDED
    if only in ("both", "guided"):
        pipe = build_pipeline(model_path, dtype, memory_profile)
        print("[B] Generating guided...")
        _, gif_b, mp4_b = run_pass(
            pipe,
            out_prefix="metric_guided",
            seed=seed,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_step=guided_guidance_step,
            guidance_lr=guided_guidance_lr,
            travel_time=guided_travel_time,
            output_dir=output_dir,
            loss_fn=loss_fn,
            fixed_frames=fixed_frames,
            prompt=prompt,
            residual_motion_metric=residual_motion_metric,
            depth_metric_fn=depth_metric_fn,
            fused_metric_fn=fused_metric_fn,
        )
        outputs.extend([gif_b, mp4_b])
        print(f"[B] Saved: {gif_b}\n[B] Saved: {mp4_b}")
        hard_cuda_cleanup(pipe)

    print("\nDone. Outputs:")
    for p in outputs:
        print(f"  - {p}")

# ---------------- CLI ----------------

def build_argparser():
    p = argparse.ArgumentParser(description="CogVideoX runner with residual_motion + metric_depth / metric_fused")
    p.add_argument("--model-path", type=str, required=True, help="HF id or local path (e.g., THUDM/CogVideoX-5b)")
    p.add_argument("--outdir", type=str, default="results", help="Output dir")
    p.add_argument("--seed", type=int, default=42, help="Seed")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Torch dtype")
    p.add_argument("--memory-profile", type=str, default="low", choices=["auto", "high", "medium", "low"])
    p.add_argument("--only", type=str, default="both", choices=["both", "baseline", "guided"])
    
    p.add_argument("--prompt", type=str, default="A steady 360° orbit around a detailed globe on a stand in a book-lined study.", 
                   help="Text prompt for generation. e.g., 'The camera circles a set of stacked wooden boxes under studio lighting.', 'The camera orbits slowly around one single Japanese style house.'")

    p.add_argument("--loss-fn", type=str, default="residual_motion",
                   choices=["residual_motion", "metric_depth", "metric_fused"],
                   help="Pick guidance loss.")
    p.add_argument("--fixed-frames", type=str, default="12,24,36,48",
                   help="Comma list of 1-based frame IDs (e.g., '12,24,36,48').")

    # Metric controls (shared)
    p.add_argument("--metric-grad-through-vggt", action="store_true",
                   help="Allow gradient through VGGT.")
    p.add_argument("--metric-pair-mode", type=str, default="adjacent", choices=["adjacent", "all"],
                   help="Pairs used in metric.")
    p.add_argument("--metric-ufm-scale", type=float, default=0.25,
                   help="Downscale for UFM inputs (flow is upscaled with vector scaling).")
    p.add_argument("--metric-cov-thresh", type=float, default=0.5, help="UFM covisibility threshold.")
    p.add_argument("--metric-conf-percent", type=float, default=20.0, help="VGGT confidence percentile (source).")
    p.add_argument("--metric-conf-min", type=float, default=0.2, help="VGGT confidence min floor.")
    p.add_argument("--metric-tau-z", type=float, default=0.02, help="Relative depth margin for occluder-in-front.")
    return p

def main():
    args = build_argparser().parse_args()
    run(
        model_path=args.model_path,
        output_dir=args.outdir,
        dtype=args.dtype,
        memory_profile=args.memory_profile,
        seed=args.seed,
        only=args.only,
        loss_fn=args.loss_fn,
        prompt=args.prompt,
        fixed_frames_line=args.fixed_frames,
        metric_grad_through_vggt=args.metric_grad_through_vggt,
        metric_pair_mode=args.metric_pair_mode,
        metric_ufm_scale=args.metric_ufm_scale,
        metric_cov_thresh=args.metric_cov_thresh,
        metric_conf_percent=args.metric_conf_percent,
        metric_conf_min=args.metric_conf_min,
        metric_tau_z=args.metric_tau_z,
    )

if __name__ == "__main__":
    main()