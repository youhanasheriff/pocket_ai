# Offline Vision Assistant - Technical Execution Plan

**Version:** 1.0
**Date:** February 2026
**Status:** Phase 1 - In Progress

---

## 1. Overview

This document defines the technical roadmap for building a fully offline, modular AI assistant system, beginning with a real-time obstacle detection and spoken guidance solution.

The system is designed to:

- Run on Edge devices (4GB RAM target - most likely 8GB RAM needed)
- Run on Android devices
- Run on Desktop (CPU/GPU)
- Operate fully offline
- Support modular, plug-and-play model variants

This is Phase 1 of a scalable AI framework that can later expand into additional detection and language-based capabilities.

---

## 2. Current Phase Scope (Phase 1)

The immediate objective is to deliver a production-grade offline obstacle detection assistant with:

- Live camera stream processing
- Desktop and Edge model variants
- Depth-aware obstacle estimation
- Template-based instruction engine
- Offline text-to-speech (TTS)
- Modular model loading framework

**Target end-to-end latency:** 200-300 ms.

---

## 3. High-Level System Architecture

The complete offline pipeline follows this structure:

```
Camera Stream
    |
    v
Vision Model (YOLO11 variant)
    |
    v
Depth Model (Monocular Depth Estimation)
    |
    v
Post-Processing Layer
    |
    v
Instruction Engine (Rule-Based)
    |
    v
Offline TTS Engine
    |
    v
Audio Output
```

All components operate locally without cloud dependency.

---

## 4. Model Strategy

### 4.1 Vision Model (Obstacle Detection)

The obstacle detection model has already been trained using YOLO11.

Two deployment variants will be maintained:

**Vision - Desktop Variant**

| Parameter | Value |
|:----------|:------|
| Resolution | 640 x 640 |
| Precision | FP16 |
| Target | Desktop CPU/GPU |
| Model size target | 30-60 MB |
| FPS target | 20-40 |

**Vision - Edge Variant**

| Parameter | Value |
|:----------|:------|
| Resolution | 320-416 |
| Precision | INT8 (quantized) |
| Target | 4GB Edge devices / Android |
| Model size target | < 15 MB |
| FPS target | 15-25 |

Both variants will be exported in ONNX format for cross-platform compatibility.

### 4.2 Depth Model Integration (Phase 1.5)

The current obstacle model does not estimate distance. To enable spatial guidance, a lightweight monocular depth model will be integrated.

Planned approach:
- Use a compact depth architecture (e.g., small MiDaS-type model)
- Export to ONNX
- Quantize for Edge deployment
- Fuse bounding box + depth region for distance approximation

This enables meaningful instructions such as:
> "Table detected approximately two feet ahead on your right."

### 4.3 Instruction Layer (Current Phase)

In the current phase, a **template-based rule engine** is used instead of an LLM.

Reasons:
- Lower latency
- Lower memory footprint
- Deterministic output
- Simpler debugging
- Suitable for structured scene descriptions

Example template:
```
"{object} detected {distance} {direction}."
```

### 4.4 Future Language Model Integration (Phase 2)

In later phases, a small SLM (Small Language Model) will be introduced.

Planned characteristics:
- Lightweight SLM (sub-1B or ~1B parameters)
- Fully offline capable
- Optimized via quantization (INT4 / INT8)
- Designed for contextual scene reasoning and dynamic instruction generation

### 4.5 Text-to-Speech Layer

Offline TTS will be integrated using a lightweight engine.

Targets:
- RAM usage < 200 MB
- Response time < 300 ms
- Streaming-friendly output

---

## 5. Modular Model Framework

All models follow a structured plug-and-play layout:

```
models/
  vision/
    obstacle_desktop.onnx
    obstacle_edge.onnx
  depth/
    depth_desktop.onnx
    depth_edge.onnx
  tts/
    tts_model
```

Each model package includes metadata (via `registry.json`):
- Version number
- Target hardware profile
- Precision type
- Input resolution
- Benchmark reference

The runtime automatically selects the correct variant based on device capability.

---

## 6. Performance Targets

### Edge Devices (4GB RAM)

| Metric | Target |
|:-------|:-------|
| Vision FPS | 15-25 |
| End-to-end latency | < 300 ms |
| Total RAM usage | < 2.5 GB |
| Model load time | < 2 seconds |

### Desktop

| Metric | Target |
|:-------|:-------|
| Vision FPS | 20-40 |
| End-to-end latency | < 200 ms |

---

## 7. Development Roadmap

### Phase 1.1 - Vision Stabilization
- Finalize dataset improvements
- Evaluate mAP
- Export ONNX
- Benchmark CPU + Android

### Phase 1.2 - Edge Optimization
- INT8 quantization
- Resolution tuning
- FPS and latency benchmarking

### Phase 1.3 - Instruction Engine
- Direction logic from bounding boxes
- Heuristic distance estimation
- Template-based generation

### Phase 1.4 - TTS Integration
- Integrate offline TTS
- Optimize latency

### Phase 1.5 - Depth Integration
- Integrate monocular depth model
- Fuse detection + depth
- Re-benchmark performance

### Phase 1.6 - Modular Loader System
- Model versioning
- Hardware-aware runtime selection
- Downloadable model architecture

---

## 8. Team Onboarding Plan

As additional engineers join, responsibilities separate into AI Core and Runtime layers.

### AI Core (Lead Responsibility)
- Model training and fine-tuning
- Quantization
- Export pipelines
- Accuracy benchmarking
- Depth fusion logic
- Model performance optimization

### IoT / Hardware Engineer
- Camera stream optimization
- Edge board benchmarking
- Hardware acceleration research
- Power consumption profiling

### Integration Engineer
- Android runtime integration
- Flutter plugin development
- Model download manager
- Application-level integration

---

## 9. Phase 2 Expansion Vision

After stabilizing the obstacle assistant:
- Introduce SLM-based contextual reasoning
- Add fire detection variant
- Add intruder detection variant
- Expand into a modular AI model ecosystem

---

## 10. Summary

The current focus is delivering a stable, low-latency, fully offline obstacle detection assistant with depth-aware guidance and template-based speech output.

Once validated across Edge, Android, and Desktop platforms, the architecture will evolve to support small language models and additional AI modules under a unified modular framework.
