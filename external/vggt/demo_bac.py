# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    
    gui_frame_selector = server.gui.add_text(
        "Show Points from Frames", initial_value="All"
    )
    gui_frame_selector.hint = "e.g., 'All', '5', or '1,5,10'"

    with server.gui.add_folder("Animation Controls"):
        gui_play_pause = server.gui.add_button("Pause")
        
        with server.gui.add_folder("Frame Navigation"):
            gui_prev_frame = server.gui.add_button("Previous Frame")
            gui_next_frame = server.gui.add_button("Next Frame")
            gui_first_frame = server.gui.add_button("First Frame")
            gui_last_frame = server.gui.add_button("Last Frame")
        
        gui_animation_speed = server.gui.add_slider(
            "Animation Speed (FPS)", min=0.1, max=10.0, step=0.1, initial_value=2.0
        )
        
        gui_auto_rotate = server.gui.add_checkbox("Auto Rotate Camera", initial_value=False)

    # --- CHANGE 1: Add a flag to the animation state ---
    animation_state = {
        "playing": True,
        "current_frame": 0,
        "thread": None,
        "stop_event": threading.Event(),
        "is_internal_update": False,  # Flag to prevent update callback from pausing animation
    }

    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        for f in frames: f.remove()
        frames.clear()
        for fr in frustums: fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(S)):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}", wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(), axes_length=0.05,
                axes_radius=0.002, origin_radius=0.002,
            )
            frames.append(frame_axis)
            img = (images_[img_id].transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        selection_str = gui_frame_selector.value.strip().lower()
        if selection_str == "all":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        elif not selection_str:
            frame_mask = np.zeros_like(conf_mask, dtype=bool)
        else:
            try:
                selected_indices = {int(p.strip()) for p in selection_str.split(',') if p.strip().isdigit()}
                if not selected_indices:
                    frame_mask = np.zeros_like(conf_mask, dtype=bool)
                else:
                    frame_mask = np.isin(frame_indices, list(selected_indices))
            except (ValueError, TypeError):
                frame_mask = np.zeros_like(conf_mask, dtype=bool)
        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    # --- CHANGE 2: Modify set_frame to use the flag ---
    def set_frame(frame_idx: int) -> None:
        """Set the current frame and update the visualization."""
        frame_idx = max(0, min(frame_idx, S - 1))
        animation_state["current_frame"] = frame_idx
        
        # Use the flag to signal that this update is from the animation loop
        animation_state["is_internal_update"] = True
        gui_frame_selector.value = str(frame_idx)
        animation_state["is_internal_update"] = False
        
        if gui_auto_rotate.value and frame_idx < len(frames):
            for client in server.get_clients().values():
                frame = frames[frame_idx]
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position

    def animation_loop():
        """Animation loop that runs in a separate thread."""
        while not animation_state["stop_event"].is_set():
            if animation_state["playing"]:
                current_frame = animation_state["current_frame"]
                next_frame = (current_frame + 1) % S
                set_frame(next_frame)
                sleep_time = 1.0 / gui_animation_speed.value
                time.sleep(sleep_time)
            else:
                time.sleep(0.1)

    @gui_play_pause.on_click
    def _(_) -> None:
        """Toggle play/pause for animation."""
        animation_state["playing"] = not animation_state["playing"]
        if animation_state["playing"]:
            gui_play_pause.text = "Pause"
            try:
                current_idx = int(gui_frame_selector.value)
                animation_state["current_frame"] = current_idx
            except (ValueError, TypeError):
                set_frame(0)
        else:
            gui_play_pause.text = "Play"

    def get_current_frame_from_gui() -> Optional[int]:
        try:
            return int(gui_frame_selector.value)
        except (ValueError, TypeError):
            return None

    @gui_prev_frame.on_click
    def _(_) -> None:
        current_frame = get_current_frame_from_gui()
        set_frame(current_frame - 1 if current_frame is not None else S - 1)

    @gui_next_frame.on_click
    def _(_) -> None:
        current_frame = get_current_frame_from_gui()
        set_frame(current_frame + 1 if current_frame is not None else 0)

    @gui_first_frame.on_click
    def _(_) -> None: set_frame(0)

    @gui_last_frame.on_click
    def _(_) -> None: set_frame(S - 1)

    @gui_points_conf.on_update
    def _(_) -> None: update_point_cloud()

    # --- CHANGE 3: Modify the on_update callback to check the flag ---
    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()
        current_frame = get_current_frame_from_gui()
        if current_frame is not None:
            animation_state["current_frame"] = current_frame
        
        # Only pause if the update was from the user, not the animation loop
        if animation_state["playing"] and not animation_state["is_internal_update"]:
            animation_state["playing"] = False
            gui_play_pause.text = "Play"

    @gui_show_frames.on_update
    def _(_) -> None:
        for f, fr in zip(frames, frustums):
            f.visible = gui_show_frames.value
            fr.visible = gui_show_frames.value

    visualize_frames(cam_to_world, images)

    animation_state["thread"] = threading.Thread(target=animation_loop, daemon=True)
    animation_state["thread"].start()

    print("Starting viser server...")
    if background_mode:
        def server_loop():
            while True: time.sleep(0.001)
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        try:
            while True: time.sleep(0.01)
        except KeyboardInterrupt:
            print("Shutting down...")
            animation_state["stop_event"].set()

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8888, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--max_frame",type=int, default=20)
parser.add_argument("--step",action="store_true",help="Whether to sample frames with even step to max_frame")
parser.add_argument("--clip",type=float, default=1)


def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser with frame-by-frame animation

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    model.eval()
    model = model.to(device)

    max_img_num = args.max_frame
    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    image_names.sort()

    image_names = image_names[:int(len(image_names)*args.clip)]
    if args.step:
        indices = np.linspace(0, len(image_names)-1, max_img_num, dtype=int)
        image_files_ds = [image_names[i] for i in indices]
        image_names=image_files_ds
    else:
        image_names = image_names[:max_img_num]
    print(f"Found {len(image_names)} images")
    # print(image_names[15:17])

    # print(image_names)

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast('cuda',dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()