# Pocket AI Guardian - Progress Report

**Date:** 13 February 2026
**Author:** Youhana Sheriff

---

## Summary

Got the full end-to-end pipeline running on MacBook M4: live webcam -> YOLO detection -> spatial processing -> spoken instructions + GUI preview. All offline, all real-time.

---

## What Was Done

### 1. Environment Setup
- Created `environment.yml` (conda) with Python 3.12 and all dependencies
- Installed Miniforge via Homebrew for Apple Silicon support
- Activate with: `conda activate pocket_ai`

### 2. Depth Estimation -- Implemented then Disabled
- Downloaded **Depth-Anything-V2-Small** INT8 ONNX model (26MB) to `models/depth/`
- Implemented full depth pipeline in `pipeline/depth.py`: preprocessing, ONNX inference, postprocessing to approximate meters (0.5-10m range)
- **Problem:** Depth adds ~430ms per frame, pushing total latency to ~480ms (target is <200ms)
- **Decision:** Disabled depth in orchestrator (marked TODO). Pipeline falls back to bbox-area heuristic for distance. Latency dropped to **~25ms** -- well below the 200ms target
- CoreML EP was tested but rejected (only supports 60% of graph nodes, 5x slower than CPU-only)

### 3. TTS -- Real Spoken Audio
- Initially implemented with pyttsx3, but it had issues on macOS: speech overlapping, getting cut off mid-sentence due to NSRunLoop threading problems
- Replaced with native macOS `say` command via subprocess -- reliable, no cutoff
- Uses a "latest wins" single-slot design: current utterance finishes, then only the most recent pending instruction plays next. Stale instructions are silently dropped
- "Path is clear" now respects the same 2s cooldown as other classes (was spamming every frame)

---

## Current Performance (MacBook M4)

| Metric | Value |
|:-------|:------|
| YOLO inference | ~25ms |
| Total pipeline | **~25ms** |
| FPS | ~40 |
| Latency target (<200ms) | **PASS** |

---

## Files Modified
- `environment.yml` -- new, conda env definition
- `pipeline/depth.py` -- real depth estimator (disabled)
- `pipeline/tts.py` -- macOS `say` TTS backend
- `pipeline/orchestrator.py` -- wired depth path, updated imports
- `pipeline/instructor.py` -- added cooldown to "Path is clear"
- `config.py` -- added `depth_model_path` field
- `main.py` -- auto-detects depth model, shows status in banner
- `requirements.txt` -- added pyttsx3, pyobjc, huggingface_hub
- `models/vision/registry.json` -- added depth + TTS metadata

---

## Next Steps
- Optimize depth inference to fit within latency budget (async on alternating frames, smaller input, or CoreML full-model conversion)
- Evaluate higher-quality TTS (Kokoro-82M, piper-TTS) as alternatives to macOS `say`
- ONNX export of YOLO model for faster desktop inference
