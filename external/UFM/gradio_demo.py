"""
Gradio demo for UFM models
This script provides a Gradio interface to visualize UFM outputs.
It supports both the base model and the refinement model.
This demo allows users to upload two images and see the flow visualization,
covisibility mask, and warped image.
"""

import cv2
import flow_vis
import gradio as gr
import numpy as np
import torch
from PIL import Image

# Import your model classes
from uniflowmatch.models.ufm import (
    UniFlowMatchClassificationRefinement,
    UniFlowMatchConfidence,
)
from uniflowmatch.utils.viz import warp_image_with_flow

# Global model variable
model = None
current_model_str = ""

use_gpu = torch.cuda.is_available()


def initialize_model(required_model_str=""):
    """Initialize the model - call this once at startup"""

    global model, current_model_str

    try:
        if required_model_str == "Base Model (560x420)":
            model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")
        elif required_model_str == "Refinement Model (560x420)":
            model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
        elif required_model_str == "Base Model (980x644)":
            model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base-980")
        elif required_model_str == "Refinement Model (980x644)":
            model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine-980")
        else:
            raise ValueError("Invalid model type selected.")

        # Set model to evaluation mode
        if hasattr(model, "eval"):
            model.eval()

        # Move model to GPU if available
        if use_gpu:
            print("Moving model to GPU...")
            model = model.to("cuda")

        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def process_images(source_image, target_image, model_type_choice):
    """
    Process two uploaded images and return visualizations
    """
    if source_image is None or target_image is None:
        return None, None, None, "Please upload both images."

    # Reinitialize model if type has changed
    global model, current_model_str
    if current_model_str != model_type_choice:
        print(f"Switching to {model_type_choice}...")
        model = initialize_model(model_type_choice)
        current_model_str = model_type_choice

    if model is None:
        return None, None, None, "Model not loaded. Please restart the application."

    try:
        # Convert PIL images to numpy arrays
        source_np = np.array(source_image)
        target_np = np.array(target_image)

        # Ensure images are RGB
        if len(source_np.shape) == 3 and source_np.shape[2] == 3:
            source_rgb = source_np
        else:
            source_rgb = cv2.cvtColor(source_np, cv2.COLOR_BGR2RGB)

        if len(target_np.shape) == 3 and target_np.shape[2] == 3:
            target_rgb = target_np
        else:
            target_rgb = cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB)

        print(f"Processing images with shapes: Source {source_rgb.shape}, Target {target_rgb.shape}")

        # === Predict Correspondences ===
        with torch.no_grad():
            result = model.predict_correspondences_batched(
                source_image=torch.from_numpy(source_rgb).to("cuda" if use_gpu else "cpu"),
                target_image=torch.from_numpy(target_rgb).to("cuda" if use_gpu else "cpu"),
            )

            # Extract results based on your model's output structure
            flow_output = result.flow.flow_output[0].cpu().numpy()
            covisibility = result.covisibility.mask[0].cpu().numpy()

        print(f"Flow output shape: {flow_output.shape}")
        print(f"Covisibility shape: {covisibility.shape}")

        # === Create Visualizations ===

        # 1. Flow visualization
        flow_vis_image = flow_vis.flow_to_color(flow_output.transpose(1, 2, 0))
        flow_pil = Image.fromarray(flow_vis_image.astype(np.uint8))

        # 2. Covisibility visualization - direct gray image
        covisibility_gray = (covisibility * 255).astype(np.uint8)
        covisibility_pil = Image.fromarray(covisibility_gray, mode="L")

        # 3. Warped image using actual warp function
        warped_image = warp_image_with_flow(source_rgb, None, target_rgb, flow_output.transpose(1, 2, 0))
        warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(
            warped_image
        )
        warped_image = (warped_image / 255.0).clip(0, 1)
        warped_pil = Image.fromarray((warped_image * 255).astype(np.uint8))

        status_msg = f"Processing completed with {model_type_choice}"

        return flow_pil, covisibility_pil, warped_pil, status_msg

    except Exception as e:
        error_msg = f"Error processing images: {str(e)}"
        print(error_msg)
        return None, None, None, error_msg


def create_demo():
    """Create the Gradio interface"""

    with gr.Blocks(title="UniFlowMatch Demo") as demo:
        gr.Markdown("# UniFlowMatch Demo")
        gr.Markdown("Upload two images to see optical flow visualization")

        # Input section
        with gr.Row():
            source_input = gr.Image(label="Source Image", type="pil")
            target_input = gr.Image(label="Target Image", type="pil")

        # Model selection
        model_type = gr.Radio(
            choices=[
                "Base Model (560x420)",
                "Refinement Model (560x420)",
                "Base Model (980x644)",
                "Refinement Model (980x644)",
            ],
            value="Base Model (560x420)",
            label="Model Type",
        )

        # Process button
        process_btn = gr.Button("Process Images")

        # Status
        status_output = gr.Textbox(label="Status", interactive=False)

        # Output section
        with gr.Row():
            flow_output = gr.Image(label="Flow Visualization")
            covisibility_output = gr.Image(label="Covisibility Mask")
            warped_output = gr.Image(label="Warped Source Image")

        # Example images
        gr.Examples(
            examples=[
                ["examples/image_pairs/fire_academy_0.png", "examples/image_pairs/fire_academy_1.png"],
                ["examples/image_pairs/scene_0.png", "examples/image_pairs/scene_1.png"],
                ["examples/image_pairs/bike_0.png", "examples/image_pairs/bike_1.png"],
                ["examples/image_pairs/cook_0.png", "examples/image_pairs/cook_1.png"],
                ["examples/image_pairs/building_0.png", "examples/image_pairs/building_1.png"],
            ],
            inputs=[source_input, target_input],
            label="Example Image Pairs",
        )

        # Event handlers
        process_btn.click(
            fn=process_images,
            inputs=[source_input, target_input, model_type],
            outputs=[flow_output, covisibility_output, warped_output, status_output],
        )

        # Auto-process when both images are uploaded
        def auto_process(source, target, model_choice):
            if source is not None and target is not None:
                return process_images(source, target, model_choice)
            return None, None, None, "Upload both images to start processing."

        for input_component in [source_input, target_input, model_type]:
            input_component.change(
                fn=auto_process,
                inputs=[source_input, target_input, model_type],
                outputs=[flow_output, covisibility_output, warped_output, status_output],
            )

    return demo


def main():
    """Main function for running the Gradio demo."""
    # Initialize model
    print("Initializing UniFlowMatch model...")
    model_loaded = initialize_model(required_model_str="Base Model (560x420)")  # Start with base model
    current_model_str = "Base Model (560x420)"

    if not model_loaded:
        print("Error: Model failed to load. Please check your model installation and HuggingFace access.")
        print("Make sure you have:")
        print("1. Installed uniflowmatch package")
        print("2. Have internet access for downloading pretrained models")
        print("3. All required dependencies installed")
        exit(1)

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",  # Allow external connections
        server_port=7860,  # Default Gradio port
        show_error=True,
    )


if __name__ == "__main__":
    main()
