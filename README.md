# Pocket AI Guardian 👁️🔊

**Offline, real-time obstacle-detection assistant that turns a camera feed into spoken navigation guidance — built to run on the edge.**

Pocket AI Guardian is a modular, fully on-device pipeline that detects navigation hazards (obstacles, doors, stairs, people, …) from a live camera and speaks concise, prioritized guidance. It runs without a network connection and is tuned to the latency and memory budgets of edge hardware (RDK X5 / Qualcomm QCS6490) as well as desktops.

> **Pipeline:** Camera → YOLO11 detection → spatial reasoning → template instructions → TTS / console

## Highlights
- **Real-time on-device inference** — YOLO11 detector with a **desktop profile** (FP16/FP32, 640px, 20–40 FPS target) and an **edge profile** (INT8, 320px, 15–25 FPS target), selected automatically or via `--profile`.
- **Spatial reasoning** — turns raw detections into distance/direction context and prioritizes by safety level.
- **Optional depth estimation** — INT8 ONNX depth model, with a heuristic fallback for lower latency.
- **Spoken guidance** — offline text-to-speech using a "latest-wins" single-slot model with deduplication, priority, and a cooldown so the user isn't overwhelmed.
- **Fully offline** — no network calls at inference time.

## Detection classes
`closed_door` · `door` · `elevator` · `escalator` · `footpath` · `obstacle` · `person` · `wall`

Each class maps to a **safety level** (high / medium / low) that drives alert priority — e.g. `obstacle`, `person`, and `escalator` are high-priority.

## Architecture
```
camera → detector (YOLO11) → spatial → validator → instructor → tts / console
                                ↑
                          depth (ONNX, optional)
```

| Module | Role |
|---|---|
| `pipeline/detector.py` | YOLO11 object detection |
| `pipeline/depth.py` | Depth estimation (INT8 ONNX + heuristic fallback) |
| `pipeline/spatial.py` | Distance / direction reasoning |
| `pipeline/validator.py` | Safety-rule validation |
| `pipeline/instructor.py` | Template-based instruction generation |
| `pipeline/tts.py` | Offline text-to-speech (latest-wins, dedup, cooldown) |
| `pipeline/orchestrator.py` | Wires the pipeline together |
| `config.py` | Single source of truth: hardware profiles, classes, thresholds |

## Quick start
```bash
pip install -r requirements.txt

python main.py                      # webcam, auto-detect hardware
python main.py --source 0 --show    # webcam with OpenCV preview
python main.py --source image.jpg   # single image
python main.py --profile edge       # force edge profile (INT8, 320px)
python main.py --profile desktop    # force desktop profile
python main.py --list-models        # show available models
python main.py --conf 0.35          # custom confidence threshold
python main.py --cooldown 3.0       # instruction cooldown (seconds)
python main.py --save-logs          # save JSONL detection logs to logs/
```

## Tech stack
- **Vision:** Ultralytics YOLO11, OpenCV, NumPy
- **Edge inference:** ONNX Runtime (INT8 models)
- **TTS:** pyttsx3 (offline)
- **Hardware targets:** Desktop (Mac/PC), RDK X5 (10 TOPS BPU), Qualcomm QCS6490 (12 TOPS NPU)
- **Bundled models:** `models/vision/obstacle_desktop.pt` (YOLO11), `models/depth/depth_small_int8.onnx`

## Tests
```bash
pytest tests/
```

## Docs
See [`docs/`](docs/) for the architecture overview, deployment guide, technical execution plan, and training report.

---
*Part of an edge-AI research line exploring assistive, real-time AI on resource-constrained hardware. See also [models-edge-devices](https://github.com/youhanasheriff/models-edge-devices).*
