"""
Utility functions for visualization.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def warp_image_with_flow(source_image, source_mask, target_image, flow) -> np.ndarray:
    """
    Warp the target to source image using the given flow vectors.
    Flow vectors indicate the displacement from source to target.

    Args:
    source_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    target_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    flow: np.ndarray of shape (H, W, 2)
    source_mask: non_occluded mask represented in source image.

    Returns:
    warped_image: target_image warped according to flow into frame of source image
    np.ndarray of shape (H, W, 3), normalized to [0, 1]

    """
    # assert source_image.shape[-1] == 3
    # assert target_image.shape[-1] == 3

    assert flow.shape[-1] == 2

    # Get the shape of the source image
    height, width = source_image.shape[:2]
    target_height, target_width = target_image.shape[:2]

    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply flow displacements
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    x_new = np.clip(x + flow_x, 0, target_width - 1) + 0.5
    y_new = np.clip(y + flow_y, 0, target_height - 1) + 0.5

    x_new = (x_new / target_image.shape[1]) * 2 - 1
    y_new = (y_new / target_image.shape[0]) * 2 - 1

    warped_image = F.grid_sample(
        torch.from_numpy(target_image).permute(2, 0, 1)[None, ...].float(),
        torch.from_numpy(np.stack([x_new, y_new], axis=-1)).float()[None, ...],
        mode="bilinear",
        align_corners=False,
    )

    warped_image = warped_image[0].permute(1, 2, 0).numpy()

    if source_mask is not None:
        warped_image = warped_image * (source_mask > 0.5)

    return warped_image


def visualize_flow(flow, flow_scale):
    """
    Visualize optical flow with direction modulating color and magnitude modulating saturation in HSV color space.

    Args:
        flow (np.ndarray): Flow array of shape (H, W, 2), where the first dimension
                           represents (flow_x, flow_y).
        flow_scale (float): The scaling factor for the magnitude of the flow.

    Returns:
        np.ndarray: An RGB image visualizing the flow.
    """
    # Convert CHW to HWC
    flow_hwc = flow

    # Compute the magnitude and angle of the flow
    magnitude = np.sqrt(np.square(flow_hwc[..., 0]) + np.square(flow_hwc[..., 1]))
    angle = np.arctan2(flow_hwc[..., 1], flow_hwc[..., 0])  # Angle in radians (-pi, pi)

    # Normalize the magnitude with the provided flow scale
    magnitude = magnitude / flow_scale
    magnitude = np.clip(magnitude, 0, 1)  # Clip values to [0, 1] for saturation

    # Convert angle from radians to degrees (used for color hue in HSV)
    angle_deg = np.degrees(angle) % 360  # Convert angle to [0, 360] degrees

    # Create an HSV image: hue is based on angle, saturation on magnitude, and value is always 1
    hsv_image = np.zeros((flow_hwc.shape[0], flow_hwc.shape[1], 3), dtype=np.uint8)
    hsv_image[..., 0] = angle_deg / 2  # OpenCV expects hue in range [0, 180]
    hsv_image[..., 1] = magnitude * 255  # Saturation in range [0, 255]
    hsv_image[..., 2] = 255  # Value always max (brightest)

    # Convert HSV image to RGB using OpenCV
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return rgb_image
