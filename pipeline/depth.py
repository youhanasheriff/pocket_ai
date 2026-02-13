"""
Depth Estimation - Depth-Anything-V2-Small via ONNX Runtime.

Provides monocular depth maps rescaled to approximate meters.
Falls back gracefully to None (triggering heuristic fallback in spatial.py)
if model is unavailable or inference fails.
"""

import logging
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Depth-Anything-V2-Small expects input aligned to 14-pixel multiples
_MODEL_SIZE = 518  # 518 = 37 * 14


class DepthEstimator:
    """
    Monocular depth estimation using Depth-Anything-V2-Small (INT8 ONNX).

    Returns depth maps as float32 HxW arrays with values in approximate meters
    (linear rescale from relative disparity into 0.5-10m range).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.available = False
        self.session = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load the ONNX model. Uses CPU EP (faster than partial CoreML for this model)."""
        try:
            import onnxruntime as ort

            # CPU is faster than CoreML for Depth-Anything-V2 because CoreML
            # only supports ~60% of the graph nodes, causing expensive EP switching.
            providers = ["CPUExecutionProvider"]

            self.session = ort.InferenceSession(
                model_path,
                providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.available = True

            active_ep = self.session.get_providers()[0]
            logger.info(f"Depth model loaded: {model_path} (EP: {active_ep})")

        except Exception as e:
            logger.warning(f"Depth model failed to load: {e}")
            self.available = False

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame -> NCHW float32 tensor normalized with ImageNet stats."""
        # BGR -> RGB and resize to model input size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_MODEL_SIZE, _MODEL_SIZE), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] then apply ImageNet mean/std
        img = resized.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC -> NCHW
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        return img.astype(np.float32)

    def _postprocess(self, raw_depth: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """
        Convert relative disparity map to approximate meters and resize to original frame.

        Depth-Anything outputs relative disparity (higher = closer).
        We invert and linearly rescale to an approximate 0.5-10m range.
        """
        depth = raw_depth.squeeze()

        # Resize to original frame dimensions
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Invert: disparity -> depth (higher disparity = closer = smaller depth)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            return np.full((orig_h, orig_w), 5.0, dtype=np.float32)

        # Normalize to [0, 1] (1 = farthest in disparity = closest in depth)
        normalized = (depth - d_min) / (d_max - d_min)

        # Invert so 0 = far, 1 = near, then map to meters
        # near (disparity=high -> normalized=1 -> inverted=0) -> 0.5m
        # far  (disparity=low  -> normalized=0 -> inverted=1) -> 10.0m
        meters = 0.5 + (1.0 - normalized) * 9.5

        return meters.astype(np.float32)

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns depth map as float32 HxW array (meters), or None if unavailable.
        """
        if not self.available or self.session is None:
            return None

        try:
            orig_h, orig_w = frame.shape[:2]
            tensor = self._preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            return self._postprocess(outputs[0], orig_h, orig_w)
        except Exception as e:
            logger.warning(f"Depth estimation failed: {e}")
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
