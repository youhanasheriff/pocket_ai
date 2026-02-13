#!/usr/bin/env python3
"""
Model Export Tool - Export YOLO11 to ONNX for deployment.

Ported from vision/yolo/export.py, simplified for pocket_ai.

Usage:
  python tools/export_model.py --profile desktop   # FP16, 640px
  python tools/export_model.py --profile edge       # INT8, 320px
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HardwareProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO11 model to ONNX")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=["desktop", "edge"],
        help="Hardware profile (desktop=FP16/640, edge=INT8/320)",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default=None,
        help="Source .pt model (default: models/vision/obstacle_desktop.pt)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    return parser.parse_args()


def check_model_size(model_path: Path, max_size_mb: float) -> dict:
    """Check if exported model meets size requirements."""
    if not model_path.exists():
        return {"exists": False}

    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return {
        "exists": True,
        "size_mb": round(size_mb, 2),
        "meets_target": size_mb <= max_size_mb,
        "target_mb": max_size_mb,
    }


def update_registry(model_name: str, size_mb: float):
    """Update registry.json with actual file size after export."""
    registry_path = MODELS_DIR / "vision" / "registry.json"
    if not registry_path.exists():
        return

    with open(registry_path) as f:
        registry = json.load(f)

    if model_name in registry.get("vision", {}):
        registry["vision"][model_name]["size_mb"] = size_mb

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")

    logger.info(f"Updated registry: {model_name} -> {size_mb}MB")


def export(args):
    from ultralytics import YOLO

    profile = HardwareProfile(args.profile)

    # Source model
    source = args.source_model or str(MODELS_DIR / "vision" / "obstacle_desktop.pt")
    source_path = Path(source)
    if not source_path.exists():
        logger.error(f"Source model not found: {source}")
        logger.error("Ensure obstacle_desktop.pt exists in models/vision/")
        sys.exit(1)

    logger.info(f"Source: {source_path}")
    logger.info(f"Profile: {profile.value}")

    # Profile-specific settings
    if profile == HardwareProfile.DESKTOP:
        img_size = 640
        half = True
        int8 = False
        output_name = "obstacle_desktop.onnx"
        registry_name = "obstacle_desktop_onnx"
        max_size_mb = 60.0
    else:
        img_size = 320
        half = False
        int8 = True
        output_name = "obstacle_edge.onnx"
        registry_name = "obstacle_edge"
        max_size_mb = 15.0

    print(f"\n{'=' * 60}")
    print(f"Exporting: {output_name}")
    print(f"  Resolution: {img_size}x{img_size}")
    print(f"  Precision: {'FP16' if half else 'INT8' if int8 else 'FP32'}")
    print(f"  Size target: <{max_size_mb}MB")
    print(f"{'=' * 60}\n")

    # Load and export
    model = YOLO(str(source_path))

    export_args = {
        "format": "onnx",
        "imgsz": img_size,
        "simplify": True,
        "opset": args.opset,
    }
    if half:
        export_args["half"] = True
    if int8:
        export_args["int8"] = True

    exported_path = Path(model.export(**export_args))

    # Copy to models/vision/
    dest_path = MODELS_DIR / "vision" / output_name
    shutil.copy2(exported_path, dest_path)

    # Check size
    size_info = check_model_size(dest_path, max_size_mb)

    if size_info["exists"]:
        status = "PASS" if size_info["meets_target"] else "WARN (exceeds target)"
        print(f"\nExported: {dest_path}")
        print(f"Size: {size_info['size_mb']:.2f} MB [{status}]")

        # Update registry
        update_registry(registry_name, size_info["size_mb"])
    else:
        logger.error("Export failed - output file not found")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Export complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    args = parse_args()
    export(args)
