"""
Pocket AI Guardian - Central Configuration

Single source of truth for all constants, thresholds, and settings.
All pipeline modules import from here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------

class HardwareProfile(Enum):
    DESKTOP = "desktop"   # Mac/PC - FP16/FP32, 640px, 20-40 FPS target
    EDGE = "edge"         # RDK X5 / QCS6490 - INT8, 320px, 15-25 FPS target


# ---------------------------------------------------------------------------
# Model classes - must match training data.yaml index order exactly
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = [
    "closed_door",   # 0
    "door",          # 1
    "elevator",      # 2
    "escalator",     # 3
    "footpath",      # 4
    "obstacle",      # 5
    "person",        # 6
    "wall",          # 7
]

# Safety level per class
# High:   immediate danger, requires alert
# Medium: potential hazard, monitor
# Low:    environmental awareness
SAFETY_LEVELS: Dict[str, str] = {
    "obstacle": "high",
    "person": "high",
    "escalator": "high",
    "door": "medium",
    "closed_door": "medium",
    "elevator": "medium",
    "footpath": "low",
    "wall": "low",
}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    profile: HardwareProfile = HardwareProfile.DESKTOP
    img_size: int = 640
    device: str = "cpu"           # "cpu" | "mps" | "cuda:0" | "0"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    max_size_mb_edge: float = 15.0
    max_size_mb_desktop: float = 60.0


@dataclass
class SpatialConfig:
    # Horizontal zones: left / center / right (thirds of frame width)
    h_zones: int = 3
    # Vertical zones: near / mid / far (thirds of frame height)
    v_zones: int = 3

    # Heuristic distance thresholds (area_ratio -> estimated meters)
    # bbox > 10% of frame area = near (~1.0m)
    area_near_threshold: float = 0.10
    # bbox 3-10% = mid (~2.5m)
    area_mid_threshold: float = 0.03
    # bbox < 3% = far (~5.0m)


@dataclass
class InstructorConfig:
    cooldown_seconds: float = 2.0         # min gap between same-class announcements
    max_instructions_per_frame: int = 2   # highest priority only
    min_confidence_to_instruct: float = 0.40


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    instructor: InstructorConfig = field(default_factory=InstructorConfig)
    show_preview: bool = False
    save_logs: bool = False
    log_dir: str = "logs/"
    target_latency_ms: int = 200   # 200 desktop / 300 edge
    depth_model_path: str = ""     # path to ONNX depth model (empty = disabled)
