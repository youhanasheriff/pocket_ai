"""
Template-Based Instruction Engine

Generates deterministic, voice-friendly instructions from enriched detections.
Replaces LLM reasoning in Phase 1 with low-latency, structured templates.

Phase 2: augmented or replaced by a small language model.
"""

import time
from typing import Dict, List, Optional

from config import InstructorConfig


# Instruction templates per class
# {direction} and {distance} are filled from spatial.description components
INSTRUCTION_TEMPLATES = {
    # High safety - immediate alerts
    "obstacle": "Warning: obstacle {direction}, {distance}. Please stop.",
    "person": "Person detected {direction}, {distance}.",
    "escalator": "Escalator {direction}, {distance}. Approach carefully.",
    # Medium safety - awareness
    "door": "Door {direction}, {distance}.",
    "closed_door": "Closed door {direction}, {distance}.",
    "elevator": "Elevator {direction}, {distance}.",
    # Low safety - contextual
    "footpath": "Footpath visible {direction}.",
    "wall": "Wall {direction}, {distance}.",
}

# Spatial zone to voice phrase mappings
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


class InstructionEngine:
    """
    Generates spoken instructions from spatially-enriched detections.

    Features:
    - Per-class cooldown to avoid repetitive announcements
    - Priority-based selection (high safety first)
    - Configurable max instructions per frame
    """

    def __init__(self, config: InstructorConfig = None):
        self.config = config or InstructorConfig()
        # Tracks last announcement time per class name
        self._cooldowns: Dict[str, float] = {}

    def _is_on_cooldown(self, class_name: str) -> bool:
        """Check if this class was announced within cooldown window."""
        last_time = self._cooldowns.get(class_name)
        if last_time is None:
            return False
        elapsed = time.monotonic() - last_time
        return elapsed < self.config.cooldown_seconds

    def _mark_announced(self, class_name: str):
        """Record announcement time for cooldown tracking."""
        self._cooldowns[class_name] = time.monotonic()

    def build_instruction(self, detection: Dict) -> Optional[str]:
        """
        Build an instruction string for a single enriched detection.

        Args:
            detection: Detection dict with 'spatial' sub-dict from SpatialProcessor

        Returns:
            Formatted instruction string, or None if on cooldown / below threshold.
        """
        cls_name = detection["class"]

        # Skip if on cooldown
        if self._is_on_cooldown(cls_name):
            return None

        # Skip if below confidence threshold
        if detection["confidence"] < self.config.min_confidence_to_instruct:
            return None

        spatial = detection.get("spatial")
        if not spatial:
            return None

        # Get template
        template = INSTRUCTION_TEMPLATES.get(cls_name)
        if not template:
            return None

        # Fill template with spatial phrases
        direction = DIRECTION_PHRASES.get(spatial["horizontal"], "ahead")
        distance = DISTANCE_PHRASES.get(spatial["distance_zone"], "nearby")

        instruction = template.format(direction=direction, distance=distance)
        return instruction

    def generate_instructions(self, detections: List[Dict]) -> List[str]:
        """
        Generate instructions for a frame's enriched detections.

        Detections are expected to already be sorted by safety priority
        (high first) from the detector.

        Args:
            detections: List of enriched detection dicts

        Returns:
            List of instruction strings (max max_instructions_per_frame).
            Returns ["Path is clear."] when no detections at all.
        """
        if not detections:
            return ["Path is clear."]

        instructions = []
        for det in detections:
            if len(instructions) >= self.config.max_instructions_per_frame:
                break

            instruction = self.build_instruction(det)
            if instruction:
                instructions.append(instruction)
                self._mark_announced(det["class"])

        return instructions

    def get_cooldown_state(self) -> Dict[str, float]:
        """Returns remaining cooldown seconds per class."""
        now = time.monotonic()
        state = {}
        for cls_name, last_time in self._cooldowns.items():
            remaining = max(0, self.config.cooldown_seconds - (now - last_time))
            if remaining > 0:
                state[cls_name] = round(remaining, 1)
        return state
