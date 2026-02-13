#!/usr/bin/env python3
"""
Pocket AI Guardian - Main Entry Point

Offline, modular AI assistant for real-time obstacle detection with spoken guidance.

Pipeline: Camera -> YOLO11n -> Spatial Processing -> Template Instructions -> Console/TTS

Usage:
  python main.py                           # webcam, auto-detect hardware
  python main.py --source 0 --show         # webcam with OpenCV preview
  python main.py --source image.jpg        # single image
  python main.py --profile edge            # force edge profile (INT8, 320px)
  python main.py --profile desktop         # force desktop profile
  python main.py --list-models             # show available models
  python main.py --conf 0.35              # custom confidence threshold
  python main.py --cooldown 3.0           # custom instruction cooldown (seconds)
  python main.py --save-logs              # save JSONL detection logs to logs/
"""

import argparse
import logging
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    HardwareProfile,
    InstructorConfig,
    ModelConfig,
    PipelineConfig,
    SpatialConfig,
)
from tools.model_loader import ModelLoader
from pipeline.orchestrator import PocketAIPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pocket AI Guardian - Offline Obstacle Detection Assistant",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (0) or image/video path",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["desktop", "edge"],
        default=None,
        help="Hardware profile (auto-detected if omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: cpu, mps, cuda:0 (auto-detected if omitted)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Detection confidence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Instruction cooldown in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames via OpenCV",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Save detection logs as JSONL to logs/",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    return parser.parse_args()


def detect_device() -> str:
    """Auto-detect the best inference device."""
    system = platform.system()

    # macOS with Apple Silicon
    if system == "Darwin":
        machine = platform.machine()
        if machine == "arm64":
            return "mps"

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass

    return "cpu"


def build_config(args) -> tuple:
    """Build PipelineConfig and resolve model path from CLI args."""
    loader = ModelLoader(PROJECT_ROOT / "models")

    # Determine hardware profile
    if args.profile:
        profile = HardwareProfile(args.profile)
    else:
        profile = loader.detect_hardware_profile()

    # Select model
    model_path, metadata = loader.select_model(profile)

    # Determine device
    device = args.device or detect_device()

    # Resolve depth model path (auto-detect if present)
    depth_model_path = ""
    depth_candidate = PROJECT_ROOT / "models" / "depth" / "depth_small_int8.onnx"
    if depth_candidate.exists():
        depth_model_path = str(depth_candidate)

    # Build config
    config = PipelineConfig(
        model=ModelConfig(
            profile=profile,
            img_size=metadata.get("img_size", 640),
            device=device,
            conf_threshold=args.conf,
        ),
        spatial=SpatialConfig(),
        instructor=InstructorConfig(
            cooldown_seconds=args.cooldown,
        ),
        show_preview=args.show,
        save_logs=args.save_logs,
        depth_model_path=depth_model_path,
    )

    return config, model_path


def main():
    args = parse_args()

    # List models mode
    if args.list_models:
        loader = ModelLoader(PROJECT_ROOT / "models")
        loader.print_models()
        return

    # Build config and resolve model
    try:
        config, model_path = build_config(args)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Print banner
    print("\n" + "=" * 55)
    print("  Pocket AI Guardian")
    print("=" * 55)
    print(f"  Profile:  {config.model.profile.value}")
    print(f"  Device:   {config.model.device}")
    print(f"  Model:    {Path(model_path).name}")
    print(f"  Img size: {config.model.img_size}px")
    print(f"  Conf:     {config.model.conf_threshold}")
    print(f"  Cooldown: {config.instructor.cooldown_seconds}s")
    print(f"  Depth:    {'enabled' if config.depth_model_path else 'disabled (heuristic)'}")
    print("=" * 55)

    # Initialize pipeline
    pipeline = PocketAIPipeline(config, model_path)

    # Determine source type and run
    source = args.source

    # Single image
    if not source.isdigit() and Path(source).is_file():
        result = pipeline.run_image(source)
        return

    # Stream (webcam or video)
    stream_source = int(source) if source.isdigit() else source
    pipeline.run_stream(
        source=stream_source,
        max_frames=args.max_frames,
        show=args.show,
    )


if __name__ == "__main__":
    main()
