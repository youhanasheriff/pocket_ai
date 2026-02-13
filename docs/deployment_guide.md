# Pocket AI - Deployment Guide

**Version:** 1.0
**Date:** February 2026

---

## Quick Start (Desktop)

```bash
cd pocket_ai/

# Install dependencies
pip install -r requirements.txt

# Verify model is ready
python main.py --list-models

# Run on webcam
python main.py --source 0

# Run on webcam with preview window
python main.py --source 0 --show

# Run on a single image
python main.py --source /path/to/image.jpg

# Run with custom settings
python main.py --source 0 --conf 0.35 --cooldown 3.0 --save-logs
```

---

## Model Export

Before deploying to edge or for faster desktop inference, export to ONNX:

```bash
# Desktop variant (FP16, 640x640)
python tools/export_model.py --profile desktop

# Edge variant (INT8, 320x320)
python tools/export_model.py --profile edge
```

After export, `main.py` will automatically prefer ONNX models over the .pt file.

---

## Hardware Profiles

### Desktop (Mac / PC / Linux workstation)

| Setting | Value |
|:--------|:------|
| Resolution | 640 x 640 |
| Precision | FP32 (.pt) or FP16 (.onnx) |
| Device | `mps` (Apple Silicon), `cuda:0` (NVIDIA), `cpu` |
| Target FPS | 20-40 |
| Target latency | < 200 ms |

```bash
python main.py --profile desktop --device mps
```

### Edge (RDK X5 / QCS6490 / Android)

| Setting | Value |
|:--------|:------|
| Resolution | 320 x 320 |
| Precision | INT8 (.onnx) |
| Device | `cpu` (or NPU via runtime) |
| Target FPS | 15-25 |
| Target latency | < 300 ms |
| Target RAM | < 2.5 GB |
| Model size | < 15 MB |

```bash
python main.py --profile edge --device cpu
```

---

## Hardware-Specific Deployment

### RDK X5 (Horizon BPU - 10 TOPS)

1. Export ONNX edge model:
   ```bash
   python tools/export_model.py --profile edge
   ```
2. Convert ONNX to Horizon BIN using the Horizon toolchain:
   ```bash
   hb_mapper makertbin --config config.yaml
   ```
3. Expected: 50+ FPS with INT8 on BPU

### Qualcomm QCS6490 (Hexagon NPU - 12 TOPS)

1. Export ONNX edge model
2. Convert using SNPE/QNN SDK:
   ```bash
   snpe-onnx-to-dlc --input_network obstacle_edge.onnx
   ```
3. Expected: 40+ FPS at 80-150mW (always-on capable)

### NVIDIA Jetson

1. Export with TensorRT engine format directly on Jetson device
2. Expected: 100+ FPS

### Android (ONNX Runtime Mobile)

1. Use the ONNX edge model (`obstacle_edge.onnx`)
2. Integrate via ONNX Runtime Android SDK
3. Expected: 15-25 FPS on mid-range devices

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --source SOURCE     Camera index (0) or image/video path (default: 0)
  --profile PROFILE   Hardware profile: desktop | edge (auto-detected)
  --device DEVICE     Inference device: cpu | mps | cuda:0 (auto-detected)
  --conf FLOAT        Detection confidence threshold (default: 0.30)
  --cooldown FLOAT    Instruction cooldown in seconds (default: 2.0)
  --show              Display annotated frames via OpenCV window
  --save-logs         Save JSONL detection logs to logs/
  --max-frames INT    Stop after N frames
  --list-models       List available models and exit
```

---

## Logging

When `--save-logs` is enabled, each session writes a JSONL file to `logs/`:

```
logs/session_1707840000.jsonl
```

Each line is a JSON object with:
```json
{
  "frame_id": 42,
  "timestamp": "2026-02-13T10:00:01Z",
  "inference_ms": 25.3,
  "total_pipeline_ms": 28.1,
  "detections_raw": 3,
  "detections_validated": 2,
  "instructions": ["Warning: obstacle directly ahead, very close. Please stop."]
}
```

---

## Performance Tuning

| Problem | Solution |
|:--------|:---------|
| FPS too low | Lower `--conf` to reduce post-processing, or use ONNX model |
| Too many repeated announcements | Increase `--cooldown` (e.g., 3.0 or 5.0) |
| Missing detections | Lower `--conf` threshold (e.g., 0.20) |
| Too many false positives | Raise `--conf` threshold (e.g., 0.40 or 0.50) |
| High memory usage | Use `--profile edge` (INT8, 320px) |
| Slow model load | Export to ONNX (avoids PyTorch startup overhead) |

---

## Project Structure

```
pocket_ai/
├── main.py                     # CLI entry point
├── config.py                   # All settings and constants
├── requirements.txt            # Python dependencies
├── docs/                       # Documentation
├── models/
│   ├── vision/
│   │   ├── obstacle_desktop.pt # Trained YOLO11n weights (5.2MB)
│   │   └── registry.json       # Model metadata
│   ├── depth/                  # Phase 1.5
│   └── tts/                    # Phase 1.4
├── pipeline/
│   ├── detector.py             # YOLO detection
│   ├── validator.py            # Detection filtering
│   ├── spatial.py              # Direction + distance zones
│   ├── instructor.py           # Template instruction engine
│   ├── tts_stub.py             # Console output (future: audio)
│   ├── depth_stub.py           # Stub (future: depth model)
│   └── orchestrator.py         # Pipeline coordinator
├── tools/
│   ├── model_loader.py         # Hardware-aware model selection
│   └── export_model.py         # ONNX export tool
└── logs/                       # Runtime detection logs
```
