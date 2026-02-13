"""
Depth Estimation Stub - Phase 1 placeholder.

Phase 1: Returns None, triggering heuristic fallback in spatial.py.
Phase 1.5: Replace with MiDaS-v3-Small or Depth-Anything-V2-Small via ONNX.
"""

from typing import Dict, Optional

import numpy as np


class DepthEstimator:
    """
    Monocular depth estimation interface.

    Phase 1: stub that always returns None.
    Phase 1.5: loads an ONNX depth model and returns real depth maps.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.available = False

        if model_path:
            # Phase 1.5: load ONNX model here
            pass

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns depth map as float32 HxW array (meters), or None if unavailable.
        """
        return None

    def get_distance_at(
        self,
        depth_map: Optional[np.ndarray],
        bbox: Dict,
    ) -> Optional[float]:
        """
        Sample median depth within bbox region of depth_map.
        Returns estimated meters or None.
        """
        if depth_map is None:
            return None

        y1 = max(0, bbox["y1"])
        y2 = min(depth_map.shape[0], bbox["y2"])
        x1 = max(0, bbox["x1"])
        x2 = min(depth_map.shape[1], bbox["x2"])

        if y2 > y1 and x2 > x1:
            region = depth_map[y1:y2, x1:x2]
            return float(np.median(region))
        return None
