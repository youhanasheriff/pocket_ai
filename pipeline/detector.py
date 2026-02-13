"""
Vision Detection Layer - Indoor Obstacle Detector

Ported from vision/yolo/inference.py.
Runs YOLO11n and outputs JSON-structured detection results.

Target: 30+ FPS real-time detection for safety applications.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from config import CLASS_NAMES, SAFETY_LEVELS

logger = logging.getLogger(__name__)


class IndoorObstacleDetector:
    """
    YOLO11-based indoor obstacle detector with JSON structured output.
    Produces detection dicts consumed by the rest of the pipeline.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        img_size: int = 640,
    ):
        from ultralytics import YOLO

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        self.class_names = CLASS_NAMES
        self.frame_count = 0

        # Performance tracking
        self.inference_times: List[float] = []
        self.max_history = 100

        logger.info(f"Detector initialized:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Confidence: {conf_threshold}")
        logger.info(f"  IoU: {iou_threshold}")
        logger.info(f"  Image size: {img_size}")

    def detect(
        self,
        source: Union[str, np.ndarray],
        return_image: bool = False,
    ) -> Dict:
        """
        Run detection on an image and return JSON-structured output.

        Args:
            source: Image path or numpy array (BGR from OpenCV)
            return_image: If True, include annotated image in output

        Returns:
            JSON-compatible dict with detections, timing, and summary.
        """
        start_time = time.perf_counter()

        results = self.model(
            source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )[0]

        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        self.frame_count += 1

        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)

        img_height, img_width = results.orig_shape
        img_area = img_height * img_width

        detections = []
        high_priority = []
        classes_detected = set()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                if cls_id < len(self.class_names):
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                classes_detected.add(cls_name)

                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                area_ratio = box_area / img_area
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                safety_level = SAFETY_LEVELS.get(cls_name, "low")
                if safety_level == "high":
                    high_priority.append(cls_name)

                detection = {
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "width": int(box_width),
                        "height": int(box_height),
                    },
                    "center": {
                        "x": int(center_x),
                        "y": int(center_y),
                    },
                    "area_ratio": round(area_ratio, 4),
                    "safety_level": safety_level,
                }
                detections.append(detection)

        # Sort by safety (high first), then confidence (descending)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        detections.sort(
            key=lambda d: (priority_order[d["safety_level"]], -d["confidence"])
        )

        # Scene complexity
        n_detections = len(detections)
        if n_detections == 0:
            scene_complexity = "clear"
        elif n_detections <= 2:
            scene_complexity = "simple"
        elif n_detections <= 5:
            scene_complexity = "moderate"
        else:
            scene_complexity = "complex"

        # FPS calculation
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        current_fps = 1000 / inference_time if inference_time > 0 else 0
        avg_fps = 1000 / avg_inference if avg_inference > 0 else 0

        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_id": self.frame_count,
            "inference_ms": round(inference_time, 2),
            "fps": {
                "current": round(current_fps, 1),
                "average": round(avg_fps, 1),
            },
            "image_size": {
                "width": img_width,
                "height": img_height,
            },
            "detections": detections,
            "summary": {
                "total_objects": n_detections,
                "high_priority": list(set(high_priority)),
                "scene_complexity": scene_complexity,
                "classes_detected": list(classes_detected),
            },
        }

        if return_image:
            output["annotated_image"] = results.plot()

        return output

    def detect_stream(
        self,
        source: Union[str, int] = 0,
        output_callback: Optional[Callable[[Dict], None]] = None,
        max_frames: Optional[int] = None,
        show: bool = False,
        save_json: Optional[str] = None,
    ):
        """
        Run detection on a video stream with JSON output per frame.

        Args:
            source: Video path or camera index (0 for webcam)
            output_callback: Called with each frame's detection dict
            max_frames: Stop after N frames (None for infinite)
            show: Display annotated frames via OpenCV
            save_json: Path to save detections as JSONL
        """
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Stream opened: {source}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}")

        json_file = None
        if save_json:
            json_file = open(save_json, "w")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of stream")
                    break

                output = self.detect(frame, return_image=show)

                if output_callback:
                    output_callback(output)

                if json_file:
                    save_output = {
                        k: v for k, v in output.items() if k != "annotated_image"
                    }
                    json_file.write(json.dumps(save_output) + "\n")

                if show and "annotated_image" in output:
                    cv2.imshow("Pocket AI - Detection", output["annotated_image"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User quit")
                        break

                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached frame limit: {max_frames}")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
            if json_file:
                json_file.close()
                logger.info(f"Saved detections to: {save_json}")

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {"frames": 0}

        times = self.inference_times
        return {
            "frames": self.frame_count,
            "avg_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "avg_fps": round(1000 / (sum(times) / len(times)), 1),
        }
