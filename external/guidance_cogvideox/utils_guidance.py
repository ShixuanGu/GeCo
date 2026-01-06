import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# --- VGGT specific imports (adjust these if your project structure differs) ---
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from vggt.utils.geometry import unproject_depth_map_to_point_map # (Not used in requested functions)

# =============================================================================
# 1. Helper Functions (Required by the main exports)
# =============================================================================

def featup(feat, target_size, source_hw=(392, 518), extra_token=5, patch_size=14, dpt_feat=False, img=False):
    """Upsamples feature maps to the target image resolution."""
    # feat_vggt: (t,h*w+5,c)
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
    
    # [B, H, W, C] -> [B, C, H, W] for interpolate
    feat = feat.permute(0, 3, 1, 2)
    
    # Interpolate each batch item (safeguard against weird shapes)
    interpolated_list = [
        F.interpolate(single_feat[None], size=(h, w), mode='bilinear', align_corners=False) 
        for single_feat in feat
    ]
    # Cat and back to [B, H, W, C]
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
    Works for single (3,3) or batched (...,3,3).
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


# =============================================================================
# 2. Main Exported Functions
# =============================================================================

def vggt_infer(
    vggt_model,
    images: torch.Tensor,                # expected shape [1, F, 3, H, W]
    upsample_size: tuple | None = None,  # (H_ref, W_ref) or None
    point_prediction: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    enable_grad: bool = False,           # <-- NEW: turn on to allow gradients
):
    if device is not None:
        images = images.to(device, non_blocking=True)

    use_cuda = images.is_cuda
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
        if use_cuda else torch.autocast(device_type="cpu", dtype=compute_dtype)
    )

    # IMPORTANT: do not wrap with torch.no_grad() unless enable_grad is False.
    # This controls whether depth/pose become part of the autograd graph w.r.t inputs.
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
            depth_map = depth_map.squeeze(0)          # [F?, Hd, Wd, 1] or [Hd,Wd,1] depending on model
            depth_conf = depth_conf.squeeze(0)[..., None]

            # (4) optional upsample depth + scale K to match upsample_size
            if upsample_size is not None:
                src_img_hw = tuple(map(int, images.shape[-2:]))        # (Hs, Ws)
                # make sure update_intrinsics uses pure torch math (no numpy) to keep graph if you need it
                # if you don't want grads through K/pose, you can call this in no_grad as it doesn't affect depth grads
                intrinsic = update_intrinsics(intrinsic, src_img_hw, upsample_size)

                src_depth_hw = tuple(map(int, depth_map.shape[1:3]))   # (Hd0, Wd0)
                assert src_img_hw == src_depth_hw, f"src_img_hw:{src_img_hw} != src_depth_hw:{src_depth_hw}"

                depth_map = featup(
                    depth_map,
                    upsample_size,
                    source_hw=src_depth_hw,
                    extra_token=0,
                    dpt_feat=True
                )  # -> (B/F, H_ref, W_ref, 1)

                depth_conf = featup(
                    depth_conf,
                    upsample_size,
                    source_hw=src_depth_hw,
                    extra_token=0,
                    dpt_feat=True
                )

                if point_prediction:
                    # Only compute point map if you will use it; otherwise skip to save memory
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
                "intrinsic": intrinsic,     # grads not needed typically; fine either way
                "extrinsic": extrinsic,
                "depth_map": depth_map,     # <-- keeps grad if enable_grad=True
                "point_map": point_map,     # may be None
                "vggt_conf": depth_conf,
            }


@torch.no_grad()
def rigid_flow_from_camera_motion(
    depth1: torch.Tensor,           # (H1,W1,1), depth for frame 1
    intrinsics: torch.Tensor,       # (2,3,3) intrinsics for cam1
    extrinsics: torch.Tensor,       # (2,3,4): [ [R1|t1], [R2|t2] ]  (world->cam)
    *,
    target_size: tuple[int,int] | None = None,  # (H2, W2). If None, uses (H1,W1)
    eps: float = 1e-8
):
    """
    Compute rigid (ego-motion) optical flow from camera motion using depth1 and poses.
    Returns:
      flow:  (H1,W1,2) [du,dv] mapping pixels in view-1 to their projected positions in view-2
      valid: (H1,W1)   bool mask (depth1>0, z2>0, projection in-bounds of target_size)
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


    assert depth1.ndim == 3 and depth1.shape[2] == 1, "depth1 must be (H1,W1,1)"
    assert K1.shape == (3,3) and K2.shape == (3,3), "K1,K2 must be (3,3)"
    assert extrinsics.shape == (2, 3, 4), "extrinsics must be (2,3,4)"


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
    X1_c_flat = X1_c.view(-1, 3).T                       # (3, H1*W1)
    X2_c_flat = (R_rel @ X1_c_flat) + t_rel.view(3,1)    # (3, H1*W1)
    X2_c = X2_c_flat.T.view(H1, W1, 3)                   # (H1,W1,3)

    # Project to view-2 with K2
    proj2 = (K2 @ X2_c.view(-1,3).T)                     # (3, H1*W1)
    denom = proj2[2].clamp_min(eps)
    u2 = (proj2[0] / denom).view(H1, W1)
    v2 = (proj2[1] / denom).view(H1, W1)

    # Original coords
    u1 = uu
    v1 = vv

    # Flow
    flow = torch.stack([u2 - u1, v2 - v1], dim=-1)       # (H1,W1,2)

    # Validity: positive depth in cam1 & cam2 and in-bounds in target image
    z1_valid = Z1 > 0
    z2_valid = X2_c[..., 2] > 0
    inb = (u2 >= 0) & (u2 <= (Wt - 1)) & (v2 >= 0) & (v2 <= (Ht - 1))
    valid = z1_valid & z2_valid & inb

    # Zero-out invalids (convenience)
    flow = torch.where(valid.unsqueeze(-1), flow, torch.zeros_like(flow))

    return flow, valid


def normalize_flow_to_unitless(
    flow_residual: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Converts the components of a flow residual from pixels to a unitless
    angular error, preserving the (H, W, 2) shape.

    Args:
        flow_residual (torch.Tensor): The (H, W, 2) tensor of pixel errors.
        intrinsics (torch.Tensor): The (3, 3) camera intrinsics matrix K.

    Returns:
        torch.Tensor: An (H, W, 2) tensor of the unitless angular
                      error components [error_u, error_v].
    """
    focal_lengths = torch.tensor(
        [intrinsics[0, 0], intrinsics[1, 1]],
        device=flow_residual.device,
        dtype=flow_residual.dtype
    )

    # This is the key step: division makes the components unitless
    angular_error_components = flow_residual / focal_lengths

    return angular_error_components


def create_confidence_mask_torch(conf: torch.Tensor, percentile_val: float = 50.0, min_threshold: float = 0.2) -> torch.Tensor:
    """
    Creates a confidence mask from a 2D tensor.

    The mask is true where the confidence is both above a calculated
    percentile and a fixed minimum threshold.

    Args:
        conf: A 2D (h, w) PyTorch tensor of confidence scores (0 to 1).
        percentile_val: The percentile to use for the adaptive threshold.
        min_threshold: The fixed minimum confidence threshold.

    Returns:
        A 2D boolean tensor of the same shape as conf.
    """
    # The percentile function in Torch requires a 1D tensor, so we flatten it.
    adaptive_threshold = torch.quantile(conf.flatten(), q=percentile_val / 100.0)
    
    # Create the final mask by combining the two conditions
    mask = (conf >= adaptive_threshold) & (conf > min_threshold)
    
    return mask


def _ufm_flow_with_grad(ufm_model, src_img01: torch.Tensor, tgt_img01: torch.Tensor):
    """
    src_img01, tgt_img01: [H,W,3] float in [0,1], requires_grad=True expected
    Returns:
      flow: [H,W,2] (requires_grad=True if inputs require grad)
      cov:  [H,W]
    """
    # Make BHWC float32
    if src_img01.dim() == 3:
        s = src_img01.unsqueeze(0)  # [1,H,W,3] or [1,3,H,W]
    else:
        s = src_img01
    if s.shape[-1] == 3:
        s_bhwc = s
    else:  # CHW -> HWC
        s_bhwc = s.permute(0, 2, 3, 1).contiguous()
    s_bhwc = s_bhwc.to(dtype=torch.float32)

    if tgt_img01.dim() == 3:
        t = tgt_img01.unsqueeze(0)
    else:
        t = tgt_img01
    if t.shape[-1] == 3:
        t_bhwc = t
    else:
        t_bhwc = t.permute(0, 2, 3, 1).contiguous()
    t_bhwc = t_bhwc.to(dtype=torch.float32)

    # Ensure grad is enabled for the forward
    with torch.set_grad_enabled(s_bhwc.requires_grad or t_bhwc.requires_grad):
        out = ufm_model.predict_correspondences_batched(
            source_image=s_bhwc, target_image=t_bhwc, data_norm_type='identity'
        )

    flow = out.flow.flow_output[0].permute(1, 2, 0)  # [H,W,2]
    cov  = out.covisibility.mask[0]                  # [H,W]
    return flow, cov