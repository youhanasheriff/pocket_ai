"""
Hardware-Aware Model Loader

Selects the correct model variant based on hardware profile and available files.
Reads registry.json from the models/ directory.
"""

import json
import logging
import platform
from pathlib import Path
from typing import Dict, List, Tuple

from config import HardwareProfile

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Selects and loads the correct model based on hardware profile.

    Selection priority:
      DESKTOP -> obstacle_desktop.onnx (if exists) -> obstacle_desktop.pt
      EDGE    -> obstacle_edge.onnx (required for edge deployment)
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.registry = self.load_registry()

    def load_registry(self) -> Dict:
        """Load and parse models/vision/registry.json."""
        registry_path = self.models_dir / "vision" / "registry.json"
        if not registry_path.exists():
            logger.warning(f"Registry not found at {registry_path}")
            return {}

        with open(registry_path) as f:
            return json.load(f)

    def select_model(self, profile: HardwareProfile) -> Tuple[str, Dict]:
        """
        Select the best available model for the given profile.

        Returns:
            (absolute_model_path, metadata_dict)

        Raises:
            FileNotFoundError if no model available for the profile.
        """
        vision_models = self.registry.get("vision", {})

        # Build candidate list ordered by preference
        if profile == HardwareProfile.DESKTOP:
            candidates = ["obstacle_desktop_onnx", "obstacle_desktop"]
        else:
            candidates = ["obstacle_edge", "obstacle_desktop_onnx", "obstacle_desktop"]

        for name in candidates:
            meta = vision_models.get(name)
            if not meta:
                continue

            model_file = self.models_dir / meta["file"]
            if model_file.exists():
                logger.info(f"Selected model: {name} ({meta['format']}, {meta['precision']})")
                return str(model_file), meta

        # Nothing found
        available = self.list_available_models()
        raise FileNotFoundError(
            f"No model found for profile '{profile.value}'. "
            f"Available models: {[m['name'] for m in available if m['exists']]}. "
            f"Run 'python tools/export_model.py --profile {profile.value}' to export."
        )

    def detect_hardware_profile(self) -> HardwareProfile:
        """
        Auto-detect hardware profile.

        Heuristic:
        - Linux + aarch64 -> EDGE (likely embedded board)
        - Otherwise -> DESKTOP
        """
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "linux" and machine in ("aarch64", "armv7l"):
            logger.info(f"Detected edge platform: {system}/{machine}")
            return HardwareProfile.EDGE

        # Try psutil for RAM check
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            if ram_gb < 6:
                logger.info(f"Low RAM detected ({ram_gb:.1f}GB), using EDGE profile")
                return HardwareProfile.EDGE
        except ImportError:
            pass

        logger.info(f"Using DESKTOP profile ({system}/{machine})")
        return HardwareProfile.DESKTOP

    def list_available_models(self) -> List[Dict]:
        """List all models in registry with existence check."""
        vision_models = self.registry.get("vision", {})
        results = []

        for name, meta in vision_models.items():
            model_file = self.models_dir / meta.get("file", "")
            exists = model_file.exists()
            size_mb = None
            if exists:
                size_mb = round(model_file.stat().st_size / (1024 * 1024), 2)

            results.append({
                "name": name,
                "file": meta.get("file", ""),
                "format": meta.get("format", "unknown"),
                "precision": meta.get("precision", "unknown"),
                "img_size": meta.get("img_size"),
                "profiles": meta.get("profiles", []),
                "exists": exists,
                "size_mb": size_mb or meta.get("size_mb"),
            })

        return results

    def print_models(self):
        """Print a formatted table of available models."""
        models = self.list_available_models()

        print("\nAvailable Models:")
        print("-" * 70)
        print(f"  {'Name':<25} {'Format':<8} {'Precision':<8} {'Size':<8} {'Status'}")
        print("-" * 70)

        for m in models:
            status = "READY" if m["exists"] else "NOT EXPORTED"
            size = f"{m['size_mb']:.1f}MB" if m["size_mb"] else "---"
            print(f"  {m['name']:<25} {m['format']:<8} {m['precision']:<8} {size:<8} {status}")

        print("-" * 70)
        print(f"  Models directory: {self.models_dir}")
        print()
