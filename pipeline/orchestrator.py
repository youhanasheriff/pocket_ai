"""
Pipeline Orchestrator - Ties all modules into a single frame processing loop.

Camera -> Detect -> Validate -> Depth (stub) -> Spatial -> Instruct -> Speak

All processing is synchronous in Phase 1 for measurable, predictable latency.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from config import PipelineConfig
from pipeline.detector import IndoorObstacleDetector
from pipeline.validator import DetectionValidator
from pipeline.spatial import SpatialProcessor
from pipeline.depth_stub import DepthEstimator
from pipeline.instructor import InstructionEngine
from pipeline.tts_stub import TTSEngine

logger = logging.getLogger(__name__)


class PocketAIPipeline:
    """
    End-to-end pipeline: Camera -> Detect -> Validate -> Spatial -> Instruct -> Speak.

    Owns all module instances and implements the frame processing loop.
    """

    def __init__(self, config: PipelineConfig, model_path: str):
        self.config = config

        # Initialize all modules
        self.detector = IndoorObstacleDetector(
            model_path=model_path,
            conf_threshold=config.model.conf_threshold,
            iou_threshold=config.model.iou_threshold,
            device=config.model.device,
            img_size=config.model.img_size,
        )
        self.validator = DetectionValidator(
            min_confidence=config.instructor.min_confidence_to_instruct,
        )
        self.spatial = SpatialProcessor(config=config.spatial)
        self.depth = DepthEstimator(
            model_path=config.depth_model_path or None,
        )
        self.instructor = InstructionEngine(config=config.instructor)
        self.tts = TTSEngine()

        # Pipeline-level performance tracking
        self.pipeline_times: List[float] = []
        self.max_history = 100

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the full pipeline.

        Returns:
            Pipeline result dict with detections, instructions, and timing.
        """
        pipeline_start = time.perf_counter()

        # Step 1: Detect objects
        raw = self.detector.detect(frame)

        # Step 2: Validate (filter noise, track history)
        validated = self.validator.validate(raw)

        # Step 3: Depth estimation (stub returns None -> heuristic fallback)
        depth_map = self.depth.estimate(frame)

        # Step 4: Spatial enrichment (direction + distance for each detection)
        enriched = self.spatial.enrich_frame(validated, depth_map)

        # Step 5: Generate instructions from enriched detections
        instructions = self.instructor.generate_instructions(
            enriched["detections"]
        )

        # Step 6: Speak/print instructions
        has_high_priority = any(
            d["safety_level"] == "high" for d in enriched["detections"]
        )
        for instruction in instructions:
            if instruction == "Path is clear.":
                priority = "info"
            elif has_high_priority:
                priority = "urgent"
            else:
                priority = "normal"
            self.tts.speak(instruction, priority=priority)

        # Track pipeline timing
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        self.pipeline_times.append(total_ms)
        if len(self.pipeline_times) > self.max_history:
            self.pipeline_times.pop(0)

        return {
            "frame_id": raw["frame_id"],
            "timestamp": raw["timestamp"],
            "inference_ms": raw["inference_ms"],
            "total_pipeline_ms": round(total_ms, 2),
            "detections_raw": raw["summary"]["total_objects"],
            "detections_validated": enriched["summary"]["total_objects"],
            "instructions": instructions,
        }

    def run_stream(
        self,
        source: Union[int, str] = 0,
        max_frames: Optional[int] = None,
        show: bool = False,
    ):
        """
        Run the full pipeline on a video stream.

        Args:
            source: Camera index (0) or video file path
            max_frames: Stop after N frames (None = infinite)
            show: Display annotated frames via OpenCV
        """
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"\n  Stream: {source} ({width}x{height} @ {cam_fps:.0f}fps)")
        print(f"  Press Ctrl+C to stop\n")

        # Open log file if saving
        log_file = None
        if self.config.save_logs:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"session_{int(time.time())}.jsonl"
            log_file = open(log_path, "w")
            logger.info(f"Logging to: {log_path}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of stream")
                    break

                result = self.process_frame(frame)
                frame_count += 1

                # Save log
                if log_file:
                    log_file.write(json.dumps(result) + "\n")

                # Show preview
                if show:
                    # Run detection again with image for display
                    display_result = self.detector.detect(frame, return_image=True)
                    if "annotated_image" in display_result:
                        cv2.imshow("Pocket AI", display_result["annotated_image"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("User quit")
                            break

                # Frame limit check
                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached frame limit: {max_frames}")
                    break

        except KeyboardInterrupt:
            print("\n")
            logger.info("Stopped by user")
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
            if log_file:
                log_file.close()

            self._print_stats(frame_count)

    def run_image(self, image_path: str) -> Dict:
        """Process a single image through the full pipeline."""
        import cv2

        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to read image: {image_path}")
            return {"error": f"Could not read {image_path}"}

        print(f"\n  Processing: {image_path}\n")
        result = self.process_frame(frame)

        print(f"\n  Detection: {result['detections_validated']} objects "
              f"({result['inference_ms']:.1f}ms inference, "
              f"{result['total_pipeline_ms']:.1f}ms total pipeline)")

        return result

    def _print_stats(self, frame_count: int):
        """Print session performance summary."""
        det_stats = self.detector.get_stats()

        if not self.pipeline_times:
            return

        avg_pipeline = sum(self.pipeline_times) / len(self.pipeline_times)
        target = self.config.target_latency_ms

        print("\n" + "=" * 55)
        print("  Session Summary")
        print("=" * 55)
        print(f"  Frames processed:    {frame_count}")
        print(f"  Avg inference:       {det_stats.get('avg_ms', 0):.1f} ms")
        print(f"  Avg pipeline:        {avg_pipeline:.1f} ms")
        print(f"  Avg FPS:             {det_stats.get('avg_fps', 0):.1f}")
        print(f"  Latency target:      <{target}ms "
              f"{'PASS' if avg_pipeline < target else 'FAIL'}")
        print("=" * 55 + "\n")

    def get_stats(self) -> Dict:
        """Get aggregate pipeline performance stats."""
        det_stats = self.detector.get_stats()

        if not self.pipeline_times:
            return {"frames": 0}

        return {
            "frames": det_stats.get("frames", 0),
            "inference_avg_ms": det_stats.get("avg_ms", 0),
            "inference_avg_fps": det_stats.get("avg_fps", 0),
            "pipeline_avg_ms": round(
                sum(self.pipeline_times) / len(self.pipeline_times), 2
            ),
        }
