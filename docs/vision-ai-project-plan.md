# Vision AI (Modular AI) - Project Plan
## Pocket AI Guardian — Hardware-First Edge AI Platform

---

## Project Overview

**Goal:** Build a trusted, portable AI guardian that lives in the physical world — not inside a phone or the cloud. A magnetic guardian puck with always-on vision and audio capabilities.

**Core Philosophy:** Hardware First. AI Native. Privacy by Design.

**Target Hardware:**
| Attribute | Target |
|-----------|--------|
| Platform | Qualcomm QCS6490 (or Ambarella CV board) |
| Board Size | ~40 × 40 mm |
| Thickness | <5 mm (excl. camera) |
| Weight | <20g (board only) |
| Cooling | Passive only |
| Battery | 1500-2000 mAh Li-Po |
| Runtime | 10-14 hrs mixed use, 24-48 hrs standby |
| Camera | Single 8-12 MP, 110-120° wide angle, MIPI CSI-2 |

**Team Structure:**
| Role | Lead | Members |
|------|------|---------|
| Head of Development & Planning | Youhana | — |
| Core Development/Implementation | Ryan | Youhana (AI/Architecture) |
| Hardware & IoT Integration | Jerry | Raghul, Jovin |
| Mobile/Web Integration | Marlen | — |

---

## Modes of Operation

Only **one mode runs at a time**:

| Mode | Use Case | Power Profile |
|------|----------|---------------|
| **Guardian Mode** | Child & elder safety monitoring | Always-on, 80-150 mW |
| **Assist Mode** | Blind & low-vision guidance | Session-based, 400-600 mW bursts |
| **Activity Mode** | Gym & sports tracking | User-initiated, 700-1200 mW |

**Future OTA Modes:**
- Focus & Safety Mode (posture, fatigue)
- Drive Safety Mode (driver distraction)
- Home Awareness Mode

---

## Phase 1: Foundation — Assist Mode (Obstacle Detection)

### Milestone 1.1: Vision Model Stabilization 🔄
**Status:** In Progress (YOLO11 trained, needs optimization for Qualcomm)

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Finalize obstacle detection dataset | Youhana | Enhanced training data | Week 1 |
| Evaluate mAP metrics | Youhana | mAP > 0.85 report | Week 1 |
| Export YOLO to ONNX (Qualcomm optimized) | Ryan | `obstacle_qcs.onnx` (<15MB) | Week 2 |
| INT8 quantization for QCS NPU | Ryan | Quantized model | Week 2 |
| Qualcomm runtime validation | Jerry | Working inference on QCS dev kit | Week 3 |

**Success Criteria:**
- Model size: <15 MB
- Inference: 15-25 FPS on QCS NPU
- Power: <600 mW during AI bursts

---

### Milestone 1.2: Pipeline Integration & TTS

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Port MacBook pipeline to QCS | Ryan | Running Python/C++ pipeline | Week 3 |
| Camera integration (MIPI CSI-2) | Jerry | Single camera capture @ 30 FPS | Week 3 |
| Evaluate Piper-TTS / Coqui TTS | Ryan | TTS engine selection report | Week 2 |
| Integrate offline TTS (<200MB RAM) | Ryan | Working TTS on QCS | Week 4 |
| Template-based instruction engine | Youhana | Rule-based instruction generator | Week 4 |
| Audio subsystem (mic + speaker) | Jerry | Audio I/O working | Week 4 |

**Success Criteria:**
- End-to-end: Camera → Detection → Instruction → Audio < 300ms
- TTS RAM usage < 200MB

---

### Milestone 1.3: NO Depth Integration (Design Decision)

**Note:** Depth estimation is **intentionally excluded** from the portable design.

Rationale:
- Single camera philosophy (simpler hardware, consistent AI behavior)
- Depth adds latency and power consumption
- Pose estimation and obstacle detection work well with geometry + motion vectors
- Keeps BOM cost and complexity down

**Alternative approach:**
- Use bounding box area + temporal consistency for distance heuristics
- Motion vectors for approaching/retreating detection

---

### Milestone 1.4: Modular Loader System

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Design model registry format (JSON) | Youhana | registry.json schema | Week 4 |
| Mode-scoped model loading | Ryan | Load/unload models per mode | Week 5 |
| Hardware-aware runtime selector | Ryan | Auto-optimize for QCS NPU | Week 5 |
| Model download manager (OTA) | Marlen | Secure OTA update system | Week 5 |
| Model versioning + rollback | Ryan | Version tracking | Week 6 |

---

## Phase 2: Hardware Bring-Up (Jerry + Raghul + Jovin)

### Milestone 2.1: Qualcomm QCS Dev Kit Setup

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Acquire QCS6490 dev kit | Jerry | Dev hardware in hand | Week 1 |
| Board bring-up and OS flash | Jerry | Working Linux/QNX on QCS | Week 1 |
| Validate MIPI camera interface | Jerry | 30 FPS capture verified | Week 2 |
| ISP tuning for wide-angle camera | Jerry | Camera pipeline optimized | Week 2 |
| Audio subsystem validation | Jerry | Mic + speaker working | Week 2 |
| Power profiling | Jerry | Power consumption report | Week 2 |

---

### Milestone 2.2: NPU Acceleration

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Qualcomm SNPE / QNN SDK setup | Jerry | NPU tools configured | Week 3 |
| ONNX → DLC conversion pipeline | Jerry | Working conversion flow | Week 3 |
| NPU inference benchmarking | Jerry | FPS/latency/power report | Week 4 |
| Thermal profiling (passive cooling) | Jerry | Thermal envelope validated | Week 4 |

**Success Criteria:**
- Passive cooling keeps temp < 60°C under sustained load
- No thermal throttling during 30-min continuous use

---

### Milestone 2.3: Mechanical Design

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Magnetic guardian puck enclosure | Jerry | 3D printed prototype | Week 4 |
| Magnetic dock design | Jerry | Charging dock prototype | Week 4 |
| Clip-on module (Activity Mode) | Jerry | Gym clip attachment | Week 5 |
| Tabletop base design | Jerry | Desk stand prototype | Week 5 |

---

## Phase 3: Mobile Companion App (Marlen)

### Milestone 3.1: Flutter App Framework

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Flutter project scaffold | Marlen | Base app structure | Week 2 |
| BLE pairing with QCS device | Marlen | Device discovery + pairing | Week 3 |
| Mode selection UI | Marlen | Switch between Guardian/Assist/Activity | Week 4 |
| Device status display (battery, temp) | Marlen | Real-time status UI | Week 4 |

---

### Milestone 3.2: Mode-Specific Features

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Guardian Mode: Alerts + history | Marlen | Safety event log | Week 5 |
| Assist Mode: Live audio streaming | Marlen | Audio from device to phone | Week 5 |
| Activity Mode: Workout summaries | Marlen | Rep counts, form feedback display | Week 6 |
| BLE HR monitor integration | Marlen | Heart rate sync | Week 6 |

---

### Milestone 3.3: OTA & Configuration

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Model download via app | Marlen | Push new AI models OTA | Week 6 |
| Device configuration UI | Marlen | Settings, sensitivity, modes | Week 6 |
| Usage analytics (opt-in) | Marlen | Privacy-preserving metrics | Week 6 |

---

## Phase 4: Multi-Mode AI Expansion

### Milestone 4.1: Guardian Mode (Child/Elder Safety)

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Pose estimation model (lightweight) | Youhana | Person detection + pose | Week 6-7 |
| Fall detection logic | Ryan | Fall detection algorithm | Week 7-8 |
| Inactivity detection | Ryan | "No movement" alerts | Week 8 |
| Child safety rules/heuristics | Youhana | Contextual safety logic | Week 8-9 |
| Elder safety rules/heuristics | Youhana | Elder-specific detection | Week 8-9 |

**Success Criteria:**
- Always-on @ 80-150 mW power
- >95% fall detection accuracy
- <5 sec alert latency

---

### Milestone 4.2: Activity Mode (Gym/Sports)

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| YOLOv8n-pose INT8 for QCS | Youhana | Pose model optimized | Week 7-8 |
| Repetition counting logic | Ryan | Accurate rep detection | Week 8-9 |
| Form correction rules | Youhana | Exercise feedback templates | Week 9-10 |
| BLE HR zone integration | Marlen | HR-based coaching | Week 9-10 |
| Clip-on mount optimization | Jerry | Gym attachment validation | Week 9 |

**Success Criteria:**
- 25-30 FPS pose estimation
- 95% repetition accuracy
- Form feedback within 1 sec

---

### Milestone 4.3: Assist Mode Enhancement

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Obstacle detection improvements | Youhana | Better accuracy, fewer false positives | Week 9-10 |
| Scene understanding (lightweight) | Youhana | Basic object classification | Week 10-11 |
| Navigation guidance templates | Youhana | Richer audio instructions | Week 10-11 |
| Voice command recognition (Whisper.cpp) | Ryan | Offline speech-to-text | Week 10-11 |

---

## Phase 5: Future OTA Modes

### Milestone 5.1: Focus & Safety Mode

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Posture detection model | Youhana | Sitting/standing posture | Week 12-13 |
| Fatigue estimation | Ryan | Eye aspect ratio + temporal patterns | Week 13-14 |
| Focus session tracking | Marlen | Productivity metrics | Week 14 |

---

### Milestone 5.2: Drive Safety Mode (Future)

| Task | Owner | Deliverable | Target |
|------|-------|-------------|--------|
| Driver gaze detection | Youhana | Eye tracking model | Week 14-16 |
| Distraction classification | Ryan | Phone use, drowsiness | Week 15-16 |
| Car mount accessory | Jerry | Vehicle mounting solution | Week 16 |

---

## Dependencies & Critical Path

### Week 1-2 Critical Path:
1. **Jerry:** Acquire and bring up QCS dev kit
2. **Youhana:** Finalize obstacle detection dataset
3. **Ryan:** Export YOLO to Qualcomm-optimized ONNX

### Hardware Blockers:
| Risk | Mitigation |
|------|------------|
| QCS dev kit unavailable | Fallback to Ambarella CV board |
| NPU compatibility issues | CPU fallback with INT8 quantization |
| Thermal throttling on passive cooling | Reduce inference frequency, optimize model |
| Camera supply chain | Lock single camera spec early |

---

## Resource Allocation

| Phase | Youhana | Ryan | Jerry | Marlen |
|-------|---------|------|-------|--------|
| Phase 1 (Weeks 1-6) | 50% | 80% | 60% | 40% |
| Phase 2 (Weeks 1-5) | 10% | 20% | 80% | 10% |
| Phase 3 (Weeks 2-6) | 10% | 10% | 20% | 100% |
| Phase 4 (Weeks 6-11) | 60% | 70% | 30% | 40% |
| Phase 5 (Weeks 12-16) | 50% | 50% | 20% | 30% |

---

## Immediate Next Steps (This Week)

### Youhana:
- [ ] Finalize obstacle detection dataset improvements
- [ ] Evaluate current mAP metrics
- [ ] Document Qualcomm QCS-specific model requirements

### Ryan:
- [ ] Research Qualcomm SNPE/QNN SDK requirements
- [ ] Prepare YOLO11 for INT8 quantization
- [ ] Set up cross-compilation environment

### Jerry:
- [ ] Source QCS6490 dev kit (or Ambarella alternative)
- [ ] Identify MIPI camera supplier (8-12 MP, 110-120°)
- [ ] Begin enclosure design for magnetic puck

### Marlen:
- [ ] Scaffold Flutter project structure
- [ ] Research BLE implementation for QCS platforms

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Obstacle Detection Latency | < 300ms end-to-end |
| Detection FPS on QCS | 15-25 FPS |
| Power (Guardian Mode) | 80-150 mW sustained |
| Power (Assist Mode bursts) | 400-600 mW |
| Battery Life (Mixed Use) | 10-14 hours |
| Battery Life (Standby) | 24-48 hours |
| Fall Detection Accuracy | > 95% |
| Rep Counting Accuracy | > 95% |
| Thermal (Passive Cooling) | < 60°C sustained |
| Model Swap Time | < 500ms |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No RDK X5 for portable** | 8-15W power, requires heatsinks, unsafe for skin contact |
| **Qualcomm QCS6490** | Ultra-low-power AI, integrated ISP/NPU/DSP, wearable-optimized |
| **Single camera (no depth)** | Simpler hardware, consistent AI behavior, lower BOM cost |
| **Passive cooling only** | Safety for wearable use, no moving parts |
| **Magnetic puck form factor** | Portable but stable, multiple mounting options |
| **One mode at a time** | Power efficiency, predictable resource usage |
| **No cloud dependency** | Privacy, latency, works offline |

---

## Notes

- **Yesterday's R&D:** Still in progress — current MacBook pipeline proves concept, needs QCS porting
- **Stereo depth:** Intentionally removed from design (see Milestone 1.3)
- **OTA strategy:** Core safety models are conservative, feature models evolve faster
- **Business model:** Hardware sold once, intelligence improves via OTA

---

*Last Updated: 2026-02-19*
*Hardware Pivot: RDK X5 → Qualcomm QCS6490*
