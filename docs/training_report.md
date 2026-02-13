# Pocket AI Guardian: Vision Model Training Completion Report

**Date:** 2026-02-09
**Model:** YOLO11 Nano (yolo11n)
**Dataset:** IndoorObstacle Detection.v9i.yolov11
**Author:** Youhana Sheriff
**Status:** Training Completed (150/150 Epochs)
**Training Run:** `indoor_obstacle_yolo11n_150ep_20260205`

---

## 1. Executive Summary

The Layer 1 perception model training for the **Pocket AI Guardian** system has been **successfully completed**. The full 150-epoch training run was executed over the weekend (Saturday 7th - Sunday 8th February 2026) on Apple Silicon hardware using Metal Performance Shaders (MPS) acceleration.

This report serves as the final completion document for the training initiative that began on **February 5th, 2026**, following the recovery from an earlier model weight deletion incident. The YOLO11n architecture has proven to be an excellent fit for edge deployment, maintaining a **5.49 MB footprint** while achieving strong detection performance with **74.72% mAP@50**.

---

## 2. Training Timeline

| Milestone | Date | Details |
|:----------|:-----|:--------|
| **Initial Training Start** | Wed, Feb 5th | Training commenced after recovery from model deletion |
| **Interim Report Issued** | Wed, Feb 5th | Progress documented at Epoch 60/150 |
| **Weekend Training Session** | Sat, Feb 7th - Sun, Feb 8th | Full dedicated training run over weekend |
| **Training Completed** | Sun, Feb 8th | All 150 epochs completed successfully |
| **Completion Report** | Mon, Feb 9th | Final documentation and results analysis |

---

## 3. Final Training Configuration

| Parameter | Value | Notes |
|:----------|:------|:------|
| **Architecture** | YOLO11n (Nano, Edge-Optimized) | Ultra-lightweight for edge devices |
| **Hardware** | Apple Silicon (M-series) | Native ARM acceleration |
| **Device** | `mps` | Metal Performance Shaders |
| **Input Resolution** | 640 x 640 | Standard detection resolution |
| **Batch Size** | 8 | Optimized from 16 to prevent OOM |
| **Total Epochs** | 150 | Full training run completed |
| **Optimizer** | AdamW | Adaptive learning rate |
| **Initial LR** | 0.001 | With cosine annealing (`cos_lr: true`) |
| **Final LR** | 0.00001 | Converged at epoch 150 |
| **Patience** | 30 epochs | Early stopping threshold |
| **AMP** | Enabled | Mixed precision training |
| **Data Augmentation** | RandAugment, Mosaic, Flip | Standard YOLO augmentation pipeline |

---

## 4. Final Performance Metrics

### 4.1 Model Performance Summary

| Metric | Initial Baseline (Feb 2) | Interim (Feb 5, Epoch 57) | **Final (Epoch 150)** | Target | Status |
|:-------|:------------------------|:-------------------------|:---------------------|:-------|:-------|
| **mAP @ 50** | 44.1% | 75.4% | **74.72%** | >70% | PASS |
| **mAP @ 50-95** | N/A | 43.9% | **44.02%** | - | Strong |
| **Precision** | 30.4% | 85.8% | **80.74%** | >50% | PASS |
| **Recall** | 56.6% | 61.5% | **68.40%** | >60% | PASS |
| **Model Size** | 5.22 MB | ~5.2 MB | **5.49 MB** | <20 MB | PASS |

### 4.2 Training Loss Progression (Epoch 150)

| Loss Component | Final Value |
|:---------------|:------------|
| **Box Loss** | 0.7375 |
| **Classification Loss** | 0.4425 |
| **DFL Loss** | 1.1862 |
| **Validation Box Loss** | 1.4480 |
| **Validation Cls Loss** | 1.1500 |
| **Validation DFL Loss** | 1.8688 |

### 4.3 Key Achievements

- 69% improvement in mAP@50 from initial baseline (44.1% -> 74.72%)
- Precision increased 165% from 30.4% to 80.74%
- Recall improved 21% from 56.6% to 68.40%
- Model size constraint maintained at 5.49 MB (well under 20 MB limit)
- All target thresholds exceeded
- Training stability achieved after batch size optimization

### 4.4 Training Convergence Analysis

The model showed strong convergence characteristics:
- **Peak mAP@50:** 80.03% at Epoch 46
- **Best Checkpoint Saved:** Based on highest validation mAP
- **Final Stabilization:** Model converged with minimal fluctuation in final 20 epochs
- **Learning Rate:** Successfully decayed from 0.001 to ~0.00001 using cosine schedule

---

## 5. Dataset

**Source:** IndoorObstacle Detection.v9i.yolov11 (Roboflow)

| Split | Images |
|:------|:-------|
| Train | 1,602 |
| Validation | 57 |

**8 Classes:**

| Class | Safety Level | Description |
|:------|:-------------|:------------|
| `obstacle` | High | Generic obstacles (furniture, objects in path) |
| `person` | High | People in the environment |
| `escalator` | High | Moving escalators |
| `door` | Medium | Open doors |
| `closed_door` | Medium | Closed doors |
| `elevator` | Medium | Elevator entrances |
| `footpath` | Low | Walkable paths |
| `wall` | Low | Wall surfaces |

---

## 6. Technical Summary

### 6.1 Challenges Overcome

| Challenge | Resolution | Outcome |
|:----------|:-----------|:--------|
| **Model Weight Deletion** | Re-initialized training from scratch | Full recovery achieved |
| **GPU Memory Overload** | Reduced batch size from 16 to 8 | Stable training throughout |
| **Hardware Thermal Management** | Weekend dedicated session with monitoring | Completed without interruption |

### 6.2 "Walking Blind Spot" Analysis

The primary challenge identified was the model's difficulty detecting persons walking directly toward the camera (ego-centric view). This was attributed to dataset bias towards side-profile static poses.

**Current status:**
- Extended training has improved overall detection robustness
- Recall improved to 68.40% indicating better detection coverage
- Independent evaluation required to assess improvement

**Ongoing mitigation - parallel data curation:**

| Data Source | Purpose | Status |
|:------------|:--------|:-------|
| **JRDB (JackRabbot Dataset)** | Robot-height perspective (~1m) | Curation in progress |
| **SCAND Dataset** | Corridor and crowd-based walking scenarios | Curation in progress |

---

## 7. Deliverables

### 7.1 Model Artifacts in pocket_ai

| Artifact | Location | Size |
|:---------|:---------|:-----|
| **obstacle_desktop.pt** | `models/vision/obstacle_desktop.pt` | 5.2 MB |
| **registry.json** | `models/vision/registry.json` | Metadata manifest |

### 7.2 Original Training Artifacts

Full training outputs are preserved at the original training location:
`models_edge_devices/vision/yolo/outputs/runs/indoor_obstacle_yolo11n_150ep_20260205/`

| Artifact | Description |
|:---------|:------------|
| `weights/best.pt` | Highest validation mAP checkpoint |
| `weights/last.pt` | Final epoch (150) checkpoint |
| `results.csv` | Complete metrics history |
| `results.png` | Training curves visualization |
| `confusion_matrix.png` | Class-wise confusion matrix |
| `confusion_matrix_normalized.png` | Normalized confusion matrix |
| `args.yaml` | Full training configuration |
| `BoxP_curve.png` | Precision curve |
| `BoxR_curve.png` | Recall curve |
| `BoxF1_curve.png` | F1 curve |
| `BoxPR_curve.png` | PR curve |

---

## 8. Next Steps

### Immediate
- [ ] Export ONNX desktop variant (FP16, 640px) via `tools/export_model.py --profile desktop`
- [ ] Export ONNX edge variant (INT8, 320px) via `tools/export_model.py --profile edge`
- [ ] Benchmark inference speed on target hardware
- [ ] Validate "Walking Blind Spot" with targeted test cases

### Integration
- [ ] Integrate depth model (Phase 1.5) for real distance estimation
- [ ] Integrate offline TTS (Phase 1.4) for audio output
- [ ] Field testing with real-world indoor scenarios

### Future Training
- [ ] Incorporate JRDB/SCAND curated data for frontal-view enhancement
- [ ] Fine-tune on ego-centric walking patterns
- [ ] Evaluate YOLO11s (Small) variant for higher accuracy on desktop

---

*Report prepared by: Youhana Sheriff*
*Date: Monday, 9th February 2026*
*Adapted for pocket_ai project: February 13th, 2026*
