# Pocket AI - Pipeline Architecture

**Version:** 1.0
**Date:** February 2026

---

## System Overview

Pocket AI runs a synchronous, modular pipeline that transforms a raw camera frame into a spoken instruction in under 200ms (desktop) / 300ms (edge).

```
Camera Frame (BGR numpy array)
       |
       v
  +-----------+     +------------+     +---------+     +------------+     +--------+
  | Detector  | --> | Validator  | --> | Spatial  | --> | Instructor | --> |  TTS   |
  | (YOLO11n) |     | (filter)   |     | (zones)  |     | (templates)|     |(speak) |
  +-----------+     +------------+     +---------+     +------------+     +--------+
       |                                    ^
       |                                    |
       |                              +----------+
       +---(future)-----------------→ |  Depth   |
                                      | Estimator|
                                      +----------+
```

All modules are orchestrated by `pipeline/orchestrator.py` which owns every instance and runs the frame loop.

---

## Module Descriptions

### 1. Detector (`pipeline/detector.py`)

**Source:** Ported from `vision/yolo/inference.py`

Runs YOLO11n inference and produces a structured JSON detection result per frame.

**Input:** BGR numpy array (640x640 desktop, 320x320 edge)
**Output:**
```json
{
  "timestamp": "2026-02-13T10:00:00Z",
  "frame_id": 42,
  "inference_ms": 25.3,
  "fps": {"current": 39.5, "average": 37.2},
  "image_size": {"width": 640, "height": 640},
  "detections": [
    {
      "class": "obstacle",
      "class_id": 5,
      "confidence": 0.85,
      "bbox": {"x1": 250, "y1": 400, "x2": 400, "y2": 600, "width": 150, "height": 200},
      "center": {"x": 325, "y": 500},
      "area_ratio": 0.073,
      "safety_level": "high"
    }
  ],
  "summary": {
    "total_objects": 1,
    "high_priority": ["obstacle"],
    "scene_complexity": "simple",
    "classes_detected": ["obstacle"]
  }
}
```

**Key details:**
- 8 classes: `closed_door`, `door`, `elevator`, `escalator`, `footpath`, `obstacle`, `person`, `wall`
- Safety levels: `high` (obstacle, person, escalator), `medium` (door, closed_door, elevator), `low` (footpath, wall)
- Detections sorted by safety priority then confidence

---

### 2. Validator (`pipeline/validator.py`)

**Source:** Adapted from `orchestration/detection_to_reasoning.py`

Filters noise and flags edge detections before spatial processing.

**Filters applied:**
- **Confidence threshold:** drops detections below `min_confidence` (default 0.40)
- **Size threshold:** drops bboxes smaller than 0.5% of frame area (noise)
- **Edge flagging:** marks detections within 5% of frame border as `edge_detection: true`

**Tracks:** Rolling frame history (last 10 frames) for temporal analysis.

---

### 3. Spatial Processor (`pipeline/spatial.py`)

Converts pixel coordinates into human-readable spatial zones.

**Horizontal zones** (frame width divided into thirds):
| Zone | Frame position | Phrase |
|:-----|:---------------|:-------|
| `left` | x < 33% | "to your left" |
| `center` | 33% < x < 66% | "directly ahead" |
| `right` | x > 66% | "to your right" |

**Distance estimation** (heuristic from bbox area ratio):
| Zone | Area ratio | Est. meters | Phrase |
|:-----|:-----------|:------------|:-------|
| `near` | > 10% | ~1.0m | "very close" |
| `mid` | 3-10% | ~2.5m | "about 2 to 3 meters ahead" |
| `far` | < 3% | ~5.0m | "ahead in the distance" |

**Output:** Adds `spatial` sub-dict to each detection:
```json
{
  "spatial": {
    "horizontal": "center",
    "vertical": "near",
    "distance_zone": "mid",
    "estimated_meters": 2.5,
    "distance_source": "heuristic",
    "description": "directly ahead, about 2 to 3 meters ahead"
  }
}
```

**Phase 1.5 upgrade:** When depth model is available, `distance_source` changes to `"depth_model"` and `estimated_meters` uses real depth values. No changes needed in downstream modules.

---

### 4. Depth Estimator (`pipeline/depth_stub.py`)

**Phase 1:** Stub that returns `None`. Spatial processor falls back to heuristic.
**Phase 1.5:** Loads MiDaS-v3-Small or Depth-Anything-V2-Small ONNX model, returns HxW float32 depth map.

The interface is stable:
```python
depth_map = depth.estimate(frame)   # None in Phase 1, np.ndarray in Phase 1.5
```

---

### 5. Instruction Engine (`pipeline/instructor.py`)

Generates voice-friendly instructions from enriched detections using templates.

**Templates:**
```
obstacle  -> "Warning: obstacle {direction}, {distance}. Please stop."
person    -> "Person detected {direction}, {distance}."
escalator -> "Escalator {direction}, {distance}. Approach carefully."
door      -> "Door {direction}, {distance}."
...
```

**Cooldown system:**
- Uses wall-clock time (`time.monotonic()`), not frame count
- Default: 2.0 seconds between repeated announcements of the same class
- Max 2 instructions per frame (highest priority first)
- Returns `["Path is clear."]` when scene is empty

**Example output:**
```
"Warning: obstacle directly ahead, very close. Please stop."
"Person detected to your right, about 2 to 3 meters ahead."
```

---

### 6. TTS Engine (`pipeline/tts_stub.py`)

**Phase 1:** Prints to console with colored priority prefixes.
```
  [ALERT] Warning: obstacle directly ahead, very close. Please stop.
  [INFO]  Person detected to your right, about 2 to 3 meters ahead.
  [NOTE]  Path is clear.
```

**Phase 1.4:** Replace internals with offline TTS backend (pyttsx3/Kokoro/piper). The `speak()` interface stays identical.

---

### 7. Orchestrator (`pipeline/orchestrator.py`)

Owns all module instances and runs the frame loop.

**`process_frame(frame)` steps:**
1. `detector.detect(frame)` - YOLO inference
2. `validator.validate(raw)` - filter noise
3. `depth.estimate(frame)` - depth map (None in Phase 1)
4. `spatial.enrich_frame(validated, depth_map)` - add zones
5. `instructor.generate_instructions(detections)` - build text
6. `tts.speak(instruction, priority)` - output

**Entry points:**
- `run_stream(source=0)` - live webcam/video loop
- `run_image(path)` - single image processing

---

## Configuration

All constants and thresholds live in `config.py`:

| Config | Controls |
|:-------|:---------|
| `ModelConfig` | device, img_size, confidence, IoU |
| `SpatialConfig` | zone count, distance thresholds |
| `InstructorConfig` | cooldown, max instructions, min confidence |
| `PipelineConfig` | combines all above + logging/preview flags |

---

## Model Registry

`models/vision/registry.json` tracks all model variants:

| Model | Format | Precision | Resolution | Status |
|:------|:-------|:----------|:-----------|:-------|
| `obstacle_desktop` | .pt | FP32 | 640 | Ready |
| `obstacle_desktop_onnx` | .onnx | FP16 | 640 | Export needed |
| `obstacle_edge` | .onnx | INT8 | 320 | Export needed |

`tools/model_loader.py` auto-selects the best model based on hardware profile (desktop vs edge).

---

## Data Flow Contract

Every module consumes and produces the same detection dict schema. Modules only *add* keys - they never remove or rename existing fields. This makes the pipeline composable and debuggable at any stage.

```
detector  -> adds: timestamp, frame_id, inference_ms, fps, detections, summary
validator -> adds: edge_detection flag, validated=true
spatial   -> adds: spatial sub-dict per detection
instructor -> consumes detections, produces instruction strings
tts        -> consumes instruction strings, produces audio/text
```
