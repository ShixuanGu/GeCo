#!/usr/bin/env python3
"""
Command-line interface for UFM.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="UFM: Unified Dense Correspondence with Flow", prog="ufm")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Launch interactive Gradio demo")
    demo_parser.add_argument("--port", type=int, default=7860, help="Port to run demo on (default: 7860)")
    demo_parser.add_argument("--share", action="store_true", help="Create public sharing link")
    demo_parser.add_argument(
        "--model", choices=["base", "refine"], default="base", help="Model variant to use (default: base)"
    )

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference on image pairs")
    infer_parser.add_argument("source", help="Source image path")
    infer_parser.add_argument("target", help="Target image path")
    infer_parser.add_argument("--output", "-o", help="Output directory (default: current directory)")
    infer_parser.add_argument(
        "--model", choices=["base", "refine"], default="base", help="Model variant to use (default: base)"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test installation")

    args = parser.parse_args()

    if args.command == "demo":
        launch_demo(args)
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "test":
        test_installation()
    else:
        parser.print_help()


def launch_demo(args):
    """Launch the Gradio demo."""
    try:
        # Import here to avoid slow startup for other commands
        from gradio_demo import create_demo, initialize_model

        print(f"Launching UFM demo with {args.model} model...")
        print(f"Demo will be available at: http://localhost:{args.port}")

        # Initialize model
        use_refinement = args.model == "refine"
        model_loaded = initialize_model(use_refinement=use_refinement)

        if not model_loaded:
            print("Error: Failed to load model. Check installation and internet connection.")
            sys.exit(1)

        # Create and launch demo
        demo = create_demo()
        demo.launch(
            share=args.share,
            server_port=args.port,
            server_name="127.0.0.1",
            show_error=True,
        )

    except ImportError as e:
        print(f"Error importing demo dependencies: {e}")
        print("Please install demo dependencies: pip install -e '.[demo]'")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching demo: {e}")
        sys.exit(1)


def run_inference(args):
    """Run inference on image pair."""
    try:
        import cv2
        import flow_vis
        import numpy as np
        import torch

        from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence
        from uniflowmatch.utils.viz import warp_image_with_flow

        # Load images
        source_img = cv2.imread(args.source)
        target_img = cv2.imread(args.target)

        if source_img is None or target_img is None:
            print("Error: Could not load one or both images")
            sys.exit(1)

        # Convert BGR to RGB
        source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Load model
        if args.model == "refine":
            model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
        else:
            model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")
        model.eval()

        print("Running inference...")

        # Run inference
        with torch.no_grad():
            result = model.predict_correspondences_batched(
                source_image=torch.from_numpy(source_rgb),
                target_image=torch.from_numpy(target_rgb),
            )

            flow = result.flow.flow_output[0].cpu().numpy()
            covisibility = result.covisibility.mask[0].cpu().numpy()

        # Save outputs
        output_dir = Path(args.output) if args.output else Path.cwd()
        output_dir.mkdir(exist_ok=True)

        # Save flow visualization
        flow_vis_img = flow_vis.flow_to_color(flow.transpose(1, 2, 0))
        cv2.imwrite(str(output_dir / "flow_visualization.png"), cv2.cvtColor(flow_vis_img, cv2.COLOR_RGB2BGR))

        # Save covisibility mask
        cv2.imwrite(str(output_dir / "covisibility_mask.png"), (covisibility * 255).astype(np.uint8))

        # Save warped image
        warped_image = warp_image_with_flow(source_rgb, None, target_rgb, flow.transpose(1, 2, 0))
        warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(
            warped_image
        )
        cv2.imwrite(str(output_dir / "warped_source.png"), cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))

        print(f"Results saved to: {output_dir}")
        print("- flow_visualization.png")
        print("- covisibility_mask.png")
        print("- warped_source.png")

    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please ensure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


def test_installation():
    """Test the installation."""
    print("Testing UFM installation...")

    try:
        # Test imports
        import torch

        print(f"✓ PyTorch {torch.__version__}")

        import torchvision

        print(f"✓ TorchVision {torchvision.__version__}")

        import numpy

        print(f"✓ NumPy {numpy.__version__}")

        import cv2

        print(f"✓ OpenCV {cv2.__version__}")

        # Test UFM imports
        from uniflowmatch.models.ufm import UniFlowMatchConfidence

        print("✓ UFM model imports")

        # Test CUDA if available
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda} available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU-only mode)")

        # Test model loading (lightweight test)
        print("Testing model loading...")
        try:
            # This should work if HuggingFace hub is accessible
            from huggingface_hub import hf_hub_download

            print("✓ HuggingFace Hub accessible")
        except Exception:
            print("⚠ HuggingFace Hub not accessible (may affect model downloading)")

        print("\n✅ Installation test completed successfully!")
        print("Run 'ufm demo' to launch the interactive demo")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check your installation")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
