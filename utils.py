import os, sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'external', 'UFM'))
sys.path.append(os.path.join(base_path, 'external', 'vggt'))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS (Dependencies for vggt_infer)
# -----------------------------------------------------------------------------

def featup(feat, target_size, source_hw=(392, 518), extra_token=5, patch_size=14, dpt_feat=False, img=False):
    """
    Upsamples feature maps to the target image resolution.
    Used internally by vggt_infer to resize depth/confidence maps.
    """
    h, w = target_size
    h_source, w_source = source_hw
    
    if extra_token > 0:
        feat = feat[:, extra_token:]
    
    if not img:
        if not dpt_feat:
            feat = feat.reshape(-1, h_source // patch_size, w_source // patch_size, feat.shape[-1])
        else:
            feat = feat.reshape(-1, h_source, w_source, feat.shape[-1])
    else:
        feat = feat.reshape(-1, h_source, w_source, feat.shape[1])
        
    feat = feat.permute(0, 3, 1, 2)
    interpolated_list = [F.interpolate(single_feat[None], size=(h, w), mode='bilinear', align_corners=False) for single_feat in feat]
    feat = torch.cat(interpolated_list, dim=0).permute(0, 2, 3, 1)
    return feat


def update_intrinsics(
    K: torch.Tensor,                 # (...,3,3)
    src_hw: Tuple[int, int],         # (Hs, Ws)
    dst_hw: Tuple[int, int],         # (Hd, Wd)
    crop: Optional[Tuple[float, float, float, float]] = None,
) -> torch.Tensor:
    """
    Adjust intrinsics for optional crop + resize.
    Used internally by vggt_infer when upsampling is requested.
    """
    Hs, Ws = map(float, src_hw)
    Hd, Wd = map(float, dst_hw)

    if crop is None:
        x0, y0, Wc, Hc = 0.0, 0.0, Ws, Hs
    else:
        x0, y0, Wc, Hc = map(float, crop)

    sx = Wd / Wc
    sy = Hd / Hc

    K_out = K.clone()

    # Focal lengths
    K_out[..., 0, 0] *= sx
    K_out[..., 1, 1] *= sy

    # Skew (rare)
    K_out[..., 0, 1] *= sx

    # Principal point
    K_out[..., 0, 2] = (K_out[..., 0, 2] - x0) * sx
    K_out[..., 1, 2] = (K_out[..., 1, 2] - y0) * sy

    return K_out


# -----------------------------------------------------------------------------
# 2. CORE FUNCTIONS
# -----------------------------------------------------------------------------

def load_image_ufm(image_path):
    """Load and preprocess an image (BGR -> RGB)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def vggt_infer(
    vggt_model,
    images: torch.Tensor,                # expected shape [1, F, 3, H, W]
    upsample_size: tuple | None = None,  # (H_ref, W_ref) or None
    point_prediction: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    enable_grad: bool = False,           
):
    """
    Runs the VGGT model to infer Camera Intrinsics, Extrinsics, Depth, and Confidence.
    """
    if device is not None:
        images = images.to(device, non_blocking=True)

    use_cuda = images.is_cuda
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
        if use_cuda else torch.autocast(device_type="cpu", dtype=compute_dtype)
    )
    
    with torch.set_grad_enabled(enable_grad):
        with autocast_ctx:
            # (1) tokens
            aggregated_tokens_list, ps_idx, _ = vggt_model.aggregator(images)

            # (2) camera
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            extrinsic, intrinsic = extrinsic.squeeze(0), intrinsic.squeeze(0)

            # (3) depth:(B,Hd,Wd,1)
            depth_map, depth_conf, _ = vggt_model.depth_head(aggregated_tokens_list, images, ps_idx)
            depth_map = depth_map.squeeze(0)          # [F?, Hd, Wd, 1] or [Hd,Wd,1]
            depth_conf = depth_conf.squeeze(0)[..., None]

            # (4) optional upsample depth + scale K to match upsample_size
            if upsample_size is not None:
                src_img_hw = tuple(map(int, images.shape[-2:]))        # (Hs, Ws)
                
                # Scale K
                intrinsic = update_intrinsics(intrinsic, src_img_hw, upsample_size)

                src_depth_hw = tuple(map(int, depth_map.shape[1:3]))   # (Hd0, Wd0)
                assert src_img_hw == src_depth_hw, f"src_img_hw:{src_img_hw} != src_depth_hw:{src_depth_hw}"

                # Upsample Depth
                depth_map = featup(
                    depth_map, upsample_size, source_hw=src_depth_hw, extra_token=0, dpt_feat=True
                )  

                # Upsample Confidence
                depth_conf = featup(
                    depth_conf, upsample_size, source_hw=src_depth_hw, extra_token=0, dpt_feat=True
                )

                if point_prediction:
                    point_map, _, _ = vggt_model.point_head(aggregated_tokens_list, images, ps_idx)
                    point_map = point_map.squeeze(0)
                    point_map = featup(
                        point_map, upsample_size, source_hw=src_img_hw, extra_token=0, dpt_feat=True
                    )
                else:
                    point_map = None
            else:
                point_map = None
                if point_prediction:
                    point_map, _, _ = vggt_model.point_head(aggregated_tokens_list, images, ps_idx)
                    point_map = point_map.squeeze(0)

            return {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "depth_map": depth_map,
                "point_map": point_map,
                "vggt_conf": depth_conf,
            }


def predict_correspondences(ufm_model, source_image, target_image, value_range="01"):
    """
    Runs the UFM model to predict 2D Optical Flow and Covisibility.
    """
    ufm_model.eval()
    for p in ufm_model.parameters():
        p.requires_grad_(False)

    device = next(ufm_model.parameters()).device

    def to_torch_hwc(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=device)
        else:
            x = x.to(device=device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4 and x.shape[1] in (1, 3):  # CHW -> HWC
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    src = to_torch_hwc(source_image).float()
    tgt = to_torch_hwc(target_image).float()
    
    if value_range == "255":
        src = src / 255.0
        tgt = tgt / 255.0
        
    with torch.enable_grad():
        out = ufm_model.predict_correspondences_batched(
            source_image=src,
            target_image=tgt,
            data_norm_type="identity",
        )

    flow = out.flow.flow_output[0].permute(1, 2, 0)  # [H,W,2]
    cov  = out.covisibility.mask[0]
    return flow, cov


def rigid_flow_from_camera_motion(
    depth1: torch.Tensor,      # (H1,W1,1), depth for frame 1
    intrinsics: torch.Tensor,  # (2,3,3) intrinsics for cam1, cam2
    extrinsics: torch.Tensor,  # (2,3,4): [ [R1|t1], [R2|t2] ]
    *,
    target_size: tuple[int,int] | None = None,
    eps: float = 1e-8
):
    """
    Compute rigid (ego-motion) optical flow from camera motion (Depth + Pose).
    Projects pixels from View 1 to View 2 based on 3D geometry.
    """
    H1, W1 = depth1.shape[:2]
    Ht, Wt = (target_size if target_size is not None else (H1, W1))

    device = depth1.device
    dtype  = depth1.dtype

    # Move params to the right device/dtype
    K1 = intrinsics[0].to(device=device, dtype=dtype)
    K2 = intrinsics[1].to(device=device, dtype=dtype)
    R1 = extrinsics[0, :, :3].to(device=device, dtype=dtype)
    t1 = extrinsics[0, :,  3].to(device=device, dtype=dtype)
    R2 = extrinsics[1, :, :3].to(device=device, dtype=dtype)
    t2 = extrinsics[1, :,  3].to(device=device, dtype=dtype)

    # cam1 -> cam2 (in camera coordinates)
    R_rel = R2 @ R1.transpose(0,1)
    t_rel = t2 - (R_rel @ t1)

    # Pixel grid in view-1 (u1,v1)
    vv, uu = torch.meshgrid(
        torch.arange(H1, device=device, dtype=dtype),
        torch.arange(W1, device=device, dtype=dtype),
        indexing="ij"
    )
    ones = torch.ones_like(uu)
    pix  = torch.stack([uu, vv, ones], dim=-1)          # (H1,W1,3)

    # Backproject to cam1 coords: X1_c = Z * K1^{-1} * [u v 1]^T
    K1_inv = torch.linalg.inv(K1)
    rays   = pix @ K1_inv.T                              # (H1,W1,3)
    Z1     = depth1.squeeze(-1)                          # (H1,W1)
    X1_c   = rays * Z1.unsqueeze(-1)                     # (H1,W1,3)

    # Transform into cam2 coords: X2_c = R_rel * X1_c + t_rel
    X1_c_flat = X1_c.view(-1, 3).T                        # (3, H1*W1)
    X2_c_flat = (R_rel @ X1_c_flat) + t_rel.view(3,1)     # (3, H1*W1)
    X2_c = X2_c_flat.T.view(H1, W1, 3)                    # (H1,W1,3)

    # Project to view-2 with K2
    proj2 = (K2 @ X2_c.view(-1,3).T)                      # (3, H1*W1)
    denom = proj2[2].clamp_min(eps)
    u2 = (proj2[0] / denom).view(H1, W1)
    v2 = (proj2[1] / denom).view(H1, W1)

    # Original coords
    u1 = uu
    v1 = vv

    # Flow
    flow = torch.stack([u2 - u1, v2 - v1], dim=-1)        # (H1,W1,2)

    # Validity: positive depth in cam1 & cam2 and in-bounds in target image
    z1_valid = Z1 > 0
    z2_valid = X2_c[..., 2] > 0
    inb = (u2 >= 0) & (u2 <= (Wt - 1)) & (v2 >= 0) & (v2 <= (Ht - 1))
    valid = z1_valid & z2_valid & inb

    # Zero-out invalids
    flow = torch.where(valid.unsqueeze(-1), flow, torch.zeros_like(flow))

    return flow, valid


def normalize_flow_to_unitless(
    flow_residual: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Converts flow residual (pixels) to unitless angular error (radians approx).
    """
    focal_lengths = torch.tensor(
        [intrinsics[0, 0], intrinsics[1, 1]],
        device=flow_residual.device,
        dtype=flow_residual.dtype
    )
    angular_error_components = flow_residual / focal_lengths
    return angular_error_components


def create_confidence_mask_torch(conf: torch.Tensor, percentile_val: float = 50.0, min_threshold: float = 0.2) -> torch.Tensor:
    """
    Creates a binary confidence mask based on percentile + min threshold.
    """
    adaptive_threshold = torch.quantile(conf.flatten(), q=percentile_val / 100.0)
    mask = (conf >= adaptive_threshold) & (conf > min_threshold)
    return mask

@torch.no_grad()
def compute_normalized_depth_error_unidirectional(
    depth_src_hw1: torch.Tensor,     # (H,W,1)  source depth D_s
    K_src_33: torch.Tensor,          # (3,3)
    E_src_34: torch.Tensor,          # (3,4) world->cam
    depth_tgt_hw1: torch.Tensor,     # (H,W,1)  target depth D_t
    K_tgt_33: torch.Tensor,          # (3,3)
    E_tgt_34: torch.Tensor,          # (3,4) world->cam
    *,
    align_corners: bool = True
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
    Xt = (R_ts @ Xs) + t_ts[:, None]
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
    D_tgt_smpl = F.grid_sample(
        D_tgt, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners
    )[0, 0]

    valid = valid & torch.isfinite(D_tgt_smpl) & (D_tgt_smpl > 0)

    Zpred  = Zt.reshape(H, W)
    deltaZ = D_tgt_smpl - Zpred
    e_z    = deltaZ.abs() / Zs

    dz_rel   = (Zpred - D_tgt_smpl) / Zs
    dz_rel_p = torch.clamp(dz_rel, min=0)
    return e_z, valid, Zs, dz_rel, dz_rel_p


@torch.no_grad()
def compute_normalized_depth_error_bidirectional(
    depth_src_hw1, K_src, E_src, depth_tgt_hw1, K_tgt, E_tgt, *, align_corners=True, occ_rel_thresh=0.02
):
    device = depth_src_hw1.device
    H, W = depth_src_hw1.shape[:2]
    Ht, Wt = depth_tgt_hw1.shape[:2]
    eps = 1e-6

    Zs = depth_src_hw1[..., 0].float().clamp(min=eps)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32), 
        torch.arange(W, device=device, dtype=torch.float32), 
        indexing='ij'
    )
    pix_s = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3).T
    Xs = (torch.inverse(K_src.float()) @ pix_s) * Zs.reshape(1, -1)

    Rs, ts = E_src[:3, :3].float(), E_src[:3, 3].float()
    Rt, tt = E_tgt[:3, :3].float(), E_tgt[:3, 3].float()
    R_ts = Rt @ Rs.T
    t_ts = tt - Rt @ Rs.T @ ts
    R_st = Rs @ Rt.T
    t_st = ts - Rs @ Rt.T @ tt

    Xt = (R_ts @ Xs) + t_ts[:, None]
    Zt_pred = Xt[2, :].clamp(min=eps)
    
    u_t = K_tgt[0, 0] * (Xt[0, :] / Zt_pred) + K_tgt[0, 2]
    v_t = K_tgt[1, 1] * (Xt[1, :] / Zt_pred) + K_tgt[1, 2]

    inside = (u_t >= 0) & (u_t <= (Wt - 1)) & (v_t >= 0) & (v_t <= (Ht - 1))
    base_valid = (inside & torch.isfinite(Zt_pred)).reshape(H, W)

    grid = torch.stack([(u_t/(Wt-1))*2-1, (v_t/(Ht-1))*2-1], dim=-1).reshape(1, H, W, 2)
    D_smpl = F.grid_sample(depth_tgt_hw1.permute(2, 0, 1).float().unsqueeze(0), grid, align_corners=align_corners)[0, 0]

    e_z = (D_smpl - Zt_pred.reshape(H,W)).abs() / Zs
    dz_rel = (Zt_pred.reshape(H,W) - D_smpl) / Zs
    mask_s2t = base_valid & torch.isfinite(D_smpl) & (D_smpl>0) & (torch.clamp(dz_rel, min=0) <= occ_rel_thresh)

    Zt = depth_tgt_hw1[..., 0].float().clamp(min=eps)
    yt, xt = torch.meshgrid(
        torch.arange(Ht, device=device, dtype=torch.float32), 
        torch.arange(Wt, device=device, dtype=torch.float32), 
        indexing='ij'
    )
    pix_t = torch.stack([xt, yt, torch.ones_like(xt)], dim=-1).reshape(-1, 3).T
    Xt_t = (torch.inverse(K_tgt.float()) @ pix_t) * Zt.reshape(1, -1)
    Xs_from_t = (R_st @ Xt_t) + t_st[:, None]
    Zs_from_t = Xs_from_t[2, :].clamp(min=eps)
    u_s = K_src[0, 0] * (Xs_from_t[0, :] / Zs_from_t) + K_src[0, 2]
    v_s = K_src[1, 1] * (Xs_from_t[1, :] / Zs_from_t) + K_src[1, 2]
    
    inside_t2s = (u_s >= 0) & (u_s <= (W - 1)) & (v_s >= 0) & (v_s <= (H - 1))
    base_valid_t2s = (inside_t2s & torch.isfinite(Zs_from_t)).reshape(Ht, Wt)
    
    grid_t2s = torch.stack([(u_s/(W-1))*2-1, (v_s/(H-1))*2-1], dim=-1).reshape(1, Ht, Wt, 2)
    D_src_smpl = F.grid_sample(depth_src_hw1.permute(2, 0, 1).float().unsqueeze(0), grid_t2s, align_corners=align_corners)[0, 0]
    dz_rel_t2s = (Zs_from_t.reshape(Ht, Wt) - D_src_smpl) / Zt
    mask_t2s_vis_tgrid = base_valid_t2s & torch.isfinite(D_src_smpl) & (D_src_smpl>0) & (torch.clamp(dz_rel_t2s, min=0) <= occ_rel_thresh)
    mask_t2s = F.grid_sample(mask_t2s_vis_tgrid.float()[None,None], grid, mode='nearest', align_corners=align_corners)[0, 0] > 0.5

    valid = mask_s2t & mask_t2s
    return e_z, valid, Zs, dz_rel, torch.clamp(dz_rel, min=0)

# -----------------------------------------------------------------------------
# 3. BENCHMARK EVAL HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.float()
    denom = m.sum()
    if denom <= 0:
        return float('nan')
    return float((x * m).sum() / denom)

# ---------------------- Time grid & windows ----------------------
def sample_indices_for_window(n_frames: int,
                              fps_native: float,
                              t_start: float,
                              t_len: float,
                              eval_fps: float):
    """
    Build a time grid inside [t_start, t_start + t_len) at eval_fps and map to nearest native frames.
    Ensures >= 2 frames.
    """
    if n_frames < 2 or t_len <= 0 or fps_native <= 0 or eval_fps <= 0:
        return []

    # Target times (exclusive end; at least 2)
    n_eval = max(2, int(np.floor(t_len * eval_fps)))
    times = t_start + (np.arange(n_eval) / eval_fps)

    # Clamp times and map to indices
    T_native = n_frames / fps_native
    times = np.clip(times, 0.0, max(0.0, T_native - 1.0 / fps_native))
    idx = np.rint(times * fps_native).astype(int)
    idx = np.clip(idx, 0, n_frames - 1)
    idx = np.unique(idx)
    if idx.size < 2:
        # Fallback to endpoints
        i0 = int(round(t_start * fps_native))
        i1 = int(round((t_start + t_len) * fps_native))
        i0 = max(0, min(n_frames - 1, i0))
        i1 = max(0, min(n_frames - 1, i1))
        idx = np.unique(np.array([i0, i1], dtype=int))
    return idx.tolist()


def build_fixed_len_cover_windows(all_files: List[str],
                                  fps_native: float,
                                  eval_fps: float,
                                  win_sec: float,
                                  max_windows: int) -> Tuple[List[List[str]], float]:
    """
    Create windows of fixed length that cover the full clip (overlap if needed).
    If keeping win_sec would exceed max_windows, enlarge window length so that
    exactly max_windows windows cover the whole clip.
    Returns: (windows, L_eff) where windows is a list of frame-file lists.
    """
    n = len(all_files)
    if n < 2:
        return [], 0.0
    T = n / float(fps_native)
    if T <= 0:
        return [], 0.0

    L = float(win_sec)
    if L <= 0:
        L_eff = T
        starts = [0.0]
    else:
        n_needed = int(math.ceil(T / L))
        if n_needed <= 1:
            L_eff = min(L, T)
            starts = [0.0]
        else:
            if n_needed <= max_windows or max_windows <= 0:
                # Keep requested win_sec; anchor last window to end; evenly space starts
                L_eff = L
                starts = [ (T - L_eff) * (k / (n_needed - 1)) for k in range(n_needed) ]
            else:
                # Cap windows but still cover clip: use exactly max_windows equal slices
                L_eff = T / float(max_windows)
                starts = [ (T - L_eff) * (k / (max_windows - 1)) for k in range(max_windows) ] if max_windows > 1 else [0.0]

    windows = []
    for s in starts:
        idx = sample_indices_for_window(n, fps_native, s, L_eff, eval_fps)
        if len(idx) >= 2:
            windows.append([all_files[i] for i in idx])

    if not windows:
        windows = [all_files]  # fallback
    return windows, L_eff

# -----------------------------------------------------------------------------
# 4. VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------
def ensure_even_dims(img):
    """Trims the image to ensure height and width are divisible by 2 (required for H.264)."""
    h, w = img.shape[:2]
    h_new = h - (h % 2)
    w_new = w - (w % 2)
    if h_new != h or w_new != w:
        return img[:h_new, :w_new]
    return img

# --- In-Memory Visualization Function ---
def render_error_maps_to_memory(
    source_image,
    *,
    motion_error_2d=None, 
    depth_error=None,     
    fused_error=None,   
    titles=("Source", "Motion Map", "Structure Map", "Fused Map"),
    dpi=100, 
    min_vmax=0.05
):
    # Helpers for applying colormap in memory
    def apply_cmap(data, cmap_name, vmin, vmax):
        if data is None: return np.zeros((100,100,3), dtype=np.uint8)
        norm = (data - vmin) / (vmax - vmin + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)
        mapper = cm.get_cmap(cmap_name)
        colored = mapper(norm)[..., :3] # Drop Alpha
        return (colored * 255).astype(np.uint8)

    def robust_vrange(arr, lo=2, hi=98, fallback=(0.0, 1.0), floor_val=min_vmax):
        if arr is None: return fallback
        vals = arr[np.isfinite(arr)]
        if vals.size == 0: return fallback
        vmin, vmax = np.percentile(vals, (lo, hi))
        if not np.isfinite(vmin): vmin = fallback[0]
        if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-6
        vmax = max(vmax, floor_val)
        return float(vmin), float(vmax)
    
    src_img = source_image.astype(np.uint8) if source_image.dtype!=np.uint8 else source_image

    # Clean inputs
    def clean(x): 
        if x is None: return None
        a = np.asarray(x).astype(np.float32)
        if a.ndim==3: a = a[...,0]
        return a
    
    mot = clean(motion_error_2d)
    dep = clean(depth_error)
    fus = clean(fused_error)
    
    # Calculate ranges
    m_vm, m_vx = robust_vrange(mot)
    d_vm, d_vx = robust_vrange(dep)
    f_vm, f_vx = robust_vrange(fus)

    # 2. Render Individual Components
    outputs = {}
    outputs["source"] = ensure_even_dims(src_img)
    outputs["motion"] = ensure_even_dims(apply_cmap(mot, 'magma', m_vm, m_vx))
    outputs["depth"]  = ensure_even_dims(apply_cmap(dep, 'viridis', d_vm, d_vx))
    outputs["fused"]  = ensure_even_dims(apply_cmap(fus, 'plasma', f_vm, f_vx))
    
    # 3. Render Grid (1x4 layout)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), dpi=dpi)
    axs = axs.ravel()
    
    plot_sequence = [
        (outputs["source"], titles[0]),
        (outputs["motion"], titles[1]),
        (outputs["depth"], titles[2]),
        (outputs["fused"], titles[3])
    ]

    for i, (data, title) in enumerate(plot_sequence):
        ax = axs[i]; ax.axis("off"); ax.set_title(title)
        if data is None: continue
        ax.imshow(data) 

    plt.tight_layout(pad=1.5)
    
    # Convert Figure to RGB Array directly in memory
    fig.canvas.draw()
    grid_rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    
    outputs["grid"] = ensure_even_dims(grid_rgb)
    return outputs