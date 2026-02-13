"""
Spatial Processing - Converts bounding boxes to human-readable spatial descriptions.

Transforms detection coordinates into directional zones (left/center/right)
and estimates distance using heuristic bbox size analysis.
Phase 1.5 will add depth model fusion.
"""

from typing import Dict, List, Optional

import numpy as np

from config import SpatialConfig


# Voice-friendly phrase mappings
DIRECTION_PHRASES = {
    "left": "to your left",
    "center": "directly ahead",
    "right": "to your right",
}

DISTANCE_PHRASES = {
    "near": "very close",
    "mid": "about 2 to 3 meters ahead",
    "far": "ahead in the distance",
}


class SpatialProcessor:
    """
    Converts YOLO detection bounding boxes to human-readable spatial descriptions.

    Adds a 'spatial' sub-dict to each detection with:
    - horizontal: left/center/right
    - vertical: near/mid/far
    - distance_zone: near/mid/far
    - estimated_meters: float (heuristic) or None (depth model)
    - distance_source: "heuristic" or "depth_model"
    - description: voice-friendly phrase
    """

    def __init__(self, config: SpatialConfig = None):
        self.config = config or SpatialConfig()

    def get_horizontal_zone(self, center_x: int, img_width: int) -> str:
        """Returns 'left', 'center', or 'right' based on x position in frame."""
        x_ratio = center_x / img_width
        if x_ratio < 1 / self.config.h_zones:
            return "left"
        elif x_ratio > (self.config.h_zones - 1) / self.config.h_zones:
            return "right"
        return "center"

    def get_vertical_zone(self, center_y: int, img_height: int) -> str:
        """
        Returns 'far', 'mid', or 'near'.
        Higher y = lower in frame = closer to camera.
        """
        y_ratio = center_y / img_height
        if y_ratio < 1 / self.config.v_zones:
            return "far"
        elif y_ratio > (self.config.v_zones - 1) / self.config.v_zones:
            return "near"
        return "mid"

    def estimate_distance_heuristic(self, area_ratio: float) -> tuple:
        """
        Heuristic distance from bbox area fraction.

        Returns:
            (zone_label, estimated_meters)
        """
        if area_ratio >= self.config.area_near_threshold:
            return "near", 1.0
        elif area_ratio >= self.config.area_mid_threshold:
            return "mid", 2.5
        return "far", 5.0

    def enrich_detection(
        self,
        detection: Dict,
        image_size: Dict,
        depth_at_bbox: Optional[float] = None,
    ) -> Dict:
        """
        Add spatial sub-dict to a single detection.

        Args:
            detection: Single detection dict from detector
            image_size: {width, height} of the source frame
            depth_at_bbox: Meters from depth model, or None for heuristic
        """
        center = detection["center"]
        h_zone = self.get_horizontal_zone(center["x"], image_size["width"])
        v_zone = self.get_vertical_zone(center["y"], image_size["height"])

        if depth_at_bbox is not None:
            # Phase 1.5: depth model provides real distance
            if depth_at_bbox < 1.5:
                distance_zone = "near"
            elif depth_at_bbox < 4.0:
                distance_zone = "mid"
            else:
                distance_zone = "far"
            estimated_meters = round(depth_at_bbox, 1)
            distance_source = "depth_model"
        else:
            # Phase 1: heuristic from bbox size
            distance_zone, estimated_meters = self.estimate_distance_heuristic(
                detection["area_ratio"]
            )
            distance_source = "heuristic"

        # Build voice-friendly description
        direction_phrase = DIRECTION_PHRASES[h_zone]
        distance_phrase = DISTANCE_PHRASES[distance_zone]
        description = f"{direction_phrase}, {distance_phrase}"

        detection["spatial"] = {
            "horizontal": h_zone,
            "vertical": v_zone,
            "distance_zone": distance_zone,
            "estimated_meters": estimated_meters,
            "distance_source": distance_source,
            "description": description,
        }

        return detection

    def enrich_frame(
        self,
        detection_json: Dict,
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Apply spatial enrichment to all detections in a frame.

        Args:
            detection_json: Validated detection dict
            depth_map: Optional HxW depth array from DepthEstimator

        Returns:
            Copy of detection_json with 'spatial' added to each detection.
        """
        enriched = detection_json.copy()
        enriched["detections"] = list(enriched["detections"])
        image_size = enriched["image_size"]

        for i, det in enumerate(enriched["detections"]):
            # Get depth value at bbox center if depth map available
            depth_at_bbox = None
            if depth_map is not None:
                bbox = det["bbox"]
                # Sample median depth within bbox region
                y1 = max(0, bbox["y1"])
                y2 = min(depth_map.shape[0], bbox["y2"])
                x1 = max(0, bbox["x1"])
                x2 = min(depth_map.shape[1], bbox["x2"])
                if y2 > y1 and x2 > x1:
                    region = depth_map[y1:y2, x1:x2]
                    depth_at_bbox = float(np.median(region))

            enriched["detections"][i] = self.enrich_detection(
                det, image_size, depth_at_bbox
            )

        return enriched
