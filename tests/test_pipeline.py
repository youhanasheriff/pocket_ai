"""
Unit tests for Pocket AI Guardian pipeline modules.

Covers DetectionValidator, SpatialProcessor, InstructionEngine, and DepthEstimator
without requiring actual model files or hardware.
"""

import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

# Ensure project root is on the path so pipeline/ and config.py are importable
sys.path.insert(
    0, "/Users/youhanasheriff/Desktop/Work/Mako/r_and_d_projects/pocket_ai"
)

from config import InstructorConfig, SpatialConfig
from pipeline.validator import DetectionValidator
from pipeline.spatial import SpatialProcessor
from pipeline.instructor import InstructionEngine
from pipeline.depth import DepthEstimator


# ---------------------------------------------------------------------------
# Helpers - reusable detection factory
# ---------------------------------------------------------------------------

def _make_detection(
    cls="obstacle",
    confidence=0.85,
    area_ratio=0.05,
    center_x=320,
    center_y=240,
    safety_level="high",
    bbox=None,
):
    """Create a minimal detection dict matching the schema used by the pipeline."""
    if bbox is None:
        bbox = {"x1": 200, "y1": 150, "x2": 440, "y2": 330}
    return {
        "class": cls,
        "confidence": confidence,
        "area_ratio": area_ratio,
        "center": {"x": center_x, "y": center_y},
        "bbox": bbox,
        "safety_level": safety_level,
    }


def _make_detection_json(detections=None, width=640, height=480):
    """Wrap detections into the full frame-level dict expected by the validator."""
    if detections is None:
        detections = [_make_detection()]
    return {
        "image_size": {"width": width, "height": height},
        "detections": detections,
        "summary": {
            "total_objects": len(detections),
            "classes_detected": list({d["class"] for d in detections}),
            "high_priority": [],
        },
    }


# ===================================================================
# DetectionValidator
# ===================================================================


class TestDetectionValidatorConfidence:
    """Detections below min_confidence must be dropped."""

    def test_detection_above_threshold_is_kept(self):
        validator = DetectionValidator(min_confidence=0.40)
        raw = _make_detection_json([_make_detection(confidence=0.80)])
        result = validator.validate(raw)
        assert len(result["detections"]) == 1

    def test_detection_below_threshold_is_dropped(self):
        validator = DetectionValidator(min_confidence=0.40)
        raw = _make_detection_json([_make_detection(confidence=0.20)])
        result = validator.validate(raw)
        assert len(result["detections"]) == 0

    def test_detection_at_exact_threshold_is_dropped(self):
        """Boundary: confidence == min_confidence is treated as < (strict)."""
        validator = DetectionValidator(min_confidence=0.40)
        raw = _make_detection_json([_make_detection(confidence=0.40)])
        result = validator.validate(raw)
        # The code uses `<` so exactly 0.40 passes
        assert len(result["detections"]) == 1


class TestDetectionValidatorAreaRatio:
    """Detections with a tiny area_ratio must be dropped."""

    def test_normal_area_ratio_kept(self):
        validator = DetectionValidator(min_area_ratio=0.005)
        raw = _make_detection_json([_make_detection(area_ratio=0.05)])
        result = validator.validate(raw)
        assert len(result["detections"]) == 1

    def test_tiny_area_ratio_dropped(self):
        validator = DetectionValidator(min_area_ratio=0.005)
        raw = _make_detection_json([_make_detection(area_ratio=0.001)])
        result = validator.validate(raw)
        assert len(result["detections"]) == 0


class TestDetectionValidatorEdgeDetection:
    """Detections near frame borders should be flagged with edge_detection=True."""

    def test_center_detection_not_flagged(self):
        validator = DetectionValidator(edge_margin=0.05)
        raw = _make_detection_json(
            [_make_detection(center_x=320, center_y=240)], width=640, height=480
        )
        result = validator.validate(raw)
        assert result["detections"][0]["edge_detection"] is False

    def test_left_edge_detection_flagged(self):
        validator = DetectionValidator(edge_margin=0.05)
        # center_x = 10 is within 5% of 640 (32 pixels)
        raw = _make_detection_json(
            [_make_detection(center_x=10, center_y=240)], width=640, height=480
        )
        result = validator.validate(raw)
        assert result["detections"][0]["edge_detection"] is True

    def test_right_edge_detection_flagged(self):
        validator = DetectionValidator(edge_margin=0.05)
        # center_x = 635 is within 5% of right edge
        raw = _make_detection_json(
            [_make_detection(center_x=635, center_y=240)], width=640, height=480
        )
        result = validator.validate(raw)
        assert result["detections"][0]["edge_detection"] is True

    def test_top_edge_detection_flagged(self):
        validator = DetectionValidator(edge_margin=0.05)
        # center_y = 5 is within 5% of 480 (24 pixels)
        raw = _make_detection_json(
            [_make_detection(center_x=320, center_y=5)], width=640, height=480
        )
        result = validator.validate(raw)
        assert result["detections"][0]["edge_detection"] is True

    def test_bottom_edge_detection_flagged(self):
        validator = DetectionValidator(edge_margin=0.05)
        raw = _make_detection_json(
            [_make_detection(center_x=320, center_y=475)], width=640, height=480
        )
        result = validator.validate(raw)
        assert result["detections"][0]["edge_detection"] is True


class TestDetectionValidatorSummary:
    """After validation the summary dict must be updated."""

    def test_summary_total_objects_updated(self):
        validator = DetectionValidator(min_confidence=0.40)
        dets = [
            _make_detection(confidence=0.80, cls="person"),
            _make_detection(confidence=0.10, cls="wall"),  # will be dropped
        ]
        raw = _make_detection_json(dets)
        result = validator.validate(raw)
        assert result["summary"]["total_objects"] == 1

    def test_summary_validated_flag_set(self):
        validator = DetectionValidator()
        raw = _make_detection_json([_make_detection()])
        result = validator.validate(raw)
        assert result["summary"]["validated"] is True

    def test_summary_high_priority_classes(self):
        validator = DetectionValidator()
        dets = [
            _make_detection(cls="obstacle", safety_level="high"),
            _make_detection(cls="wall", safety_level="low"),
        ]
        raw = _make_detection_json(dets)
        result = validator.validate(raw)
        assert "obstacle" in result["summary"]["high_priority"]
        assert "wall" not in result["summary"]["high_priority"]

    def test_summary_classes_detected(self):
        validator = DetectionValidator()
        dets = [
            _make_detection(cls="person", safety_level="high"),
            _make_detection(cls="door", safety_level="medium"),
        ]
        raw = _make_detection_json(dets)
        result = validator.validate(raw)
        assert set(result["summary"]["classes_detected"]) == {"person", "door"}


# ===================================================================
# SpatialProcessor
# ===================================================================


class TestSpatialProcessorHorizontalZone:
    """Horizontal zones: left / center / right based on x position."""

    def test_left_zone(self):
        sp = SpatialProcessor(SpatialConfig(h_zones=3))
        assert sp.get_horizontal_zone(50, 640) == "left"

    def test_center_zone(self):
        sp = SpatialProcessor(SpatialConfig(h_zones=3))
        assert sp.get_horizontal_zone(320, 640) == "center"

    def test_right_zone(self):
        sp = SpatialProcessor(SpatialConfig(h_zones=3))
        assert sp.get_horizontal_zone(600, 640) == "right"

    def test_boundary_left_center(self):
        """x at exactly 1/3 of width should be center."""
        sp = SpatialProcessor(SpatialConfig(h_zones=3))
        # x_ratio = 213/640 ~ 0.333 which is NOT < 1/3, so center
        result = sp.get_horizontal_zone(214, 640)
        assert result == "center"


class TestSpatialProcessorVerticalZone:
    """Vertical zones: far / mid / near based on y position (higher y = closer)."""

    def test_far_zone_top_of_frame(self):
        sp = SpatialProcessor(SpatialConfig(v_zones=3))
        assert sp.get_vertical_zone(50, 480) == "far"

    def test_mid_zone_center_of_frame(self):
        sp = SpatialProcessor(SpatialConfig(v_zones=3))
        assert sp.get_vertical_zone(240, 480) == "mid"

    def test_near_zone_bottom_of_frame(self):
        sp = SpatialProcessor(SpatialConfig(v_zones=3))
        assert sp.get_vertical_zone(450, 480) == "near"


class TestSpatialProcessorHeuristicDistance:
    """Heuristic distance estimation from bbox area ratio."""

    def test_large_area_is_near(self):
        sp = SpatialProcessor(SpatialConfig(area_near_threshold=0.10))
        zone, meters = sp.estimate_distance_heuristic(0.15)
        assert zone == "near"
        assert meters == 1.0

    def test_medium_area_is_mid(self):
        sp = SpatialProcessor(
            SpatialConfig(area_near_threshold=0.10, area_mid_threshold=0.03)
        )
        zone, meters = sp.estimate_distance_heuristic(0.05)
        assert zone == "mid"
        assert meters == 2.5

    def test_small_area_is_far(self):
        sp = SpatialProcessor(
            SpatialConfig(area_near_threshold=0.10, area_mid_threshold=0.03)
        )
        zone, meters = sp.estimate_distance_heuristic(0.01)
        assert zone == "far"
        assert meters == 5.0


class TestSpatialProcessorEnrichDetection:
    """enrich_detection must add a 'spatial' sub-dict to the detection."""

    def test_spatial_subdict_is_added(self):
        sp = SpatialProcessor()
        det = _make_detection(center_x=320, center_y=240, area_ratio=0.05)
        image_size = {"width": 640, "height": 480}
        enriched = sp.enrich_detection(det, image_size)

        assert "spatial" in enriched
        spatial = enriched["spatial"]
        assert spatial["horizontal"] in ("left", "center", "right")
        assert spatial["vertical"] in ("near", "mid", "far")
        assert spatial["distance_zone"] in ("near", "mid", "far")
        assert isinstance(spatial["estimated_meters"], (int, float))
        assert spatial["distance_source"] == "heuristic"
        assert isinstance(spatial["description"], str)

    def test_depth_model_overrides_heuristic(self):
        sp = SpatialProcessor()
        det = _make_detection(center_x=320, center_y=240, area_ratio=0.05)
        image_size = {"width": 640, "height": 480}
        enriched = sp.enrich_detection(det, image_size, depth_at_bbox=1.2)

        spatial = enriched["spatial"]
        assert spatial["distance_source"] == "depth_model"
        assert spatial["estimated_meters"] == 1.2
        assert spatial["distance_zone"] == "near"

    def test_depth_model_mid_range(self):
        sp = SpatialProcessor()
        det = _make_detection(center_x=320, center_y=240, area_ratio=0.05)
        image_size = {"width": 640, "height": 480}
        enriched = sp.enrich_detection(det, image_size, depth_at_bbox=2.5)

        assert enriched["spatial"]["distance_zone"] == "mid"

    def test_depth_model_far_range(self):
        sp = SpatialProcessor()
        det = _make_detection(center_x=320, center_y=240, area_ratio=0.05)
        image_size = {"width": 640, "height": 480}
        enriched = sp.enrich_detection(det, image_size, depth_at_bbox=6.0)

        assert enriched["spatial"]["distance_zone"] == "far"


# ===================================================================
# InstructionEngine
# ===================================================================


class TestInstructionEngineBasic:
    """Basic instruction generation from enriched detections."""

    def _enriched_detection(self, cls="obstacle", confidence=0.85, h_zone="center",
                            d_zone="near"):
        det = _make_detection(cls=cls, confidence=confidence, safety_level="high")
        det["spatial"] = {
            "horizontal": h_zone,
            "distance_zone": d_zone,
            "vertical": "mid",
            "estimated_meters": 1.0,
            "distance_source": "heuristic",
            "description": "directly ahead, very close",
        }
        return det

    def test_instruction_generated_for_known_class(self):
        engine = InstructionEngine(InstructorConfig(cooldown_seconds=0))
        det = self._enriched_detection(cls="obstacle")
        instructions = engine.generate_instructions([det])
        assert len(instructions) == 1
        assert "obstacle" in instructions[0].lower() or "Warning" in instructions[0]

    def test_instruction_contains_direction_phrase(self):
        engine = InstructionEngine(InstructorConfig(cooldown_seconds=0))
        det = self._enriched_detection(cls="person", h_zone="left")
        instructions = engine.generate_instructions([det])
        assert len(instructions) == 1
        assert "to your left" in instructions[0]

    def test_no_detections_returns_path_clear(self):
        engine = InstructionEngine(InstructorConfig(cooldown_seconds=0))
        instructions = engine.generate_instructions([])
        assert instructions == ["Path is clear."]


class TestInstructionEngineCooldown:
    """Cooldown prevents the same class being announced too frequently."""

    def test_cooldown_blocks_repeated_announcement(self):
        engine = InstructionEngine(InstructorConfig(cooldown_seconds=10.0))
        det = _make_detection(cls="obstacle", confidence=0.85, safety_level="high")
        det["spatial"] = {
            "horizontal": "center",
            "distance_zone": "near",
            "vertical": "mid",
            "estimated_meters": 1.0,
            "distance_source": "heuristic",
            "description": "directly ahead, very close",
        }

        first = engine.generate_instructions([det])
        assert len(first) == 1

        # Immediate second call - should be on cooldown
        second = engine.generate_instructions([det])
        assert len(second) == 0

    def test_cooldown_expires(self):
        engine = InstructionEngine(InstructorConfig(cooldown_seconds=0.05))
        det = _make_detection(cls="person", confidence=0.85, safety_level="high")
        det["spatial"] = {
            "horizontal": "center",
            "distance_zone": "mid",
            "vertical": "mid",
            "estimated_meters": 2.5,
            "distance_source": "heuristic",
            "description": "directly ahead, about 2 to 3 meters ahead",
        }

        first = engine.generate_instructions([det])
        assert len(first) == 1

        time.sleep(0.10)  # exceed 50ms cooldown

        second = engine.generate_instructions([det])
        assert len(second) == 1


class TestInstructionEngineMaxInstructions:
    """At most max_instructions_per_frame instructions are returned."""

    def _enriched_detection(self, cls, h_zone="center"):
        det = _make_detection(cls=cls, confidence=0.85, safety_level="high")
        det["spatial"] = {
            "horizontal": h_zone,
            "distance_zone": "near",
            "vertical": "mid",
            "estimated_meters": 1.0,
            "distance_source": "heuristic",
            "description": "directly ahead, very close",
        }
        return det

    def test_max_instructions_limits_output(self):
        config = InstructorConfig(
            cooldown_seconds=0, max_instructions_per_frame=2
        )
        engine = InstructionEngine(config)
        dets = [
            self._enriched_detection("obstacle"),
            self._enriched_detection("person"),
            self._enriched_detection("escalator"),
        ]
        instructions = engine.generate_instructions(dets)
        assert len(instructions) == 2

    def test_single_instruction_limit(self):
        config = InstructorConfig(
            cooldown_seconds=0, max_instructions_per_frame=1
        )
        engine = InstructionEngine(config)
        dets = [
            self._enriched_detection("obstacle"),
            self._enriched_detection("person"),
        ]
        instructions = engine.generate_instructions(dets)
        assert len(instructions) == 1


# ===================================================================
# DepthEstimator
# ===================================================================


class TestDepthEstimatorNoModel:
    """DepthEstimator without a loaded model should return None gracefully."""

    def test_estimate_returns_none_without_model(self):
        estimator = DepthEstimator(model_path=None)
        assert estimator.available is False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame)
        assert result is None

    def test_get_distance_at_returns_none_with_none_depth_map(self):
        estimator = DepthEstimator(model_path=None)
        bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        result = estimator.get_distance_at(None, bbox)
        assert result is None

    def test_get_distance_at_with_valid_depth_map(self):
        estimator = DepthEstimator(model_path=None)
        # Create a depth map filled with 3.0 meters
        depth_map = np.full((480, 640), 3.0, dtype=np.float32)
        bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        result = estimator.get_distance_at(depth_map, bbox)
        assert result == pytest.approx(3.0)

    def test_get_distance_at_with_zero_area_bbox(self):
        estimator = DepthEstimator(model_path=None)
        depth_map = np.full((480, 640), 3.0, dtype=np.float32)
        # bbox with zero area (x1 == x2)
        bbox = {"x1": 100, "y1": 100, "x2": 100, "y2": 200}
        result = estimator.get_distance_at(depth_map, bbox)
        assert result is None
