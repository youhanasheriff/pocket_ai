"""
Detection Validator - Filters noise and tracks temporal context.

Adapted from orchestration/detection_to_reasoning.py validate_detections().
"""

import logging
from collections import deque
from typing import Dict, Set

logger = logging.getLogger(__name__)


class DetectionValidator:
    """
    Validates and filters raw detection output before spatial processing.

    Applies:
    - Confidence filtering (drop below threshold)
    - Tiny detection filtering (noise from very small bboxes)
    - Edge detection flagging (partial objects near frame border)
    - Temporal frame history tracking
    """

    def __init__(
        self,
        min_confidence: float = 0.40,
        min_area_ratio: float = 0.005,
        edge_margin: float = 0.05,
        max_frame_history: int = 10,
    ):
        self.min_confidence = min_confidence
        self.min_area_ratio = min_area_ratio
        self.edge_margin = edge_margin
        self.frame_history: deque = deque(maxlen=max_frame_history)

    def validate(self, detection_json: Dict) -> Dict:
        """
        Filter detections by confidence, size, and flag edge detections.

        Args:
            detection_json: Raw output from IndoorObstacleDetector.detect()

        Returns:
            Validated detection dict (same schema, subset of detections).
        """
        validated = detection_json.copy()
        validated["detections"] = list(validated["detections"])  # don't mutate original
        img_size = validated["image_size"]

        validated_detections = []
        for det in validated["detections"]:
            # Skip low confidence
            if det["confidence"] < self.min_confidence:
                continue

            # Skip very small detections (noise)
            if det["area_ratio"] < self.min_area_ratio:
                continue

            # Flag detections near frame edges (partial objects)
            center = det["center"]
            if (
                center["x"] < img_size["width"] * self.edge_margin
                or center["x"] > img_size["width"] * (1 - self.edge_margin)
                or center["y"] < img_size["height"] * self.edge_margin
                or center["y"] > img_size["height"] * (1 - self.edge_margin)
            ):
                det["edge_detection"] = True
            else:
                det["edge_detection"] = False

            validated_detections.append(det)

        validated["detections"] = validated_detections
        validated["summary"] = dict(validated["summary"])  # don't mutate original
        validated["summary"]["total_objects"] = len(validated_detections)
        validated["summary"]["validated"] = True
        validated["summary"]["high_priority"] = list(set(
            d["class"] for d in validated_detections if d["safety_level"] == "high"
        ))
        validated["summary"]["classes_detected"] = list(set(
            d["class"] for d in validated_detections
        ))

        # Track history
        self.frame_history.append(validated)

        return validated

    def get_previous_classes(self) -> Set[str]:
        """Classes seen in any of the last N frames."""
        classes = set()
        for frame in self.frame_history:
            for det in frame.get("detections", []):
                classes.add(det["class"])
        return classes
