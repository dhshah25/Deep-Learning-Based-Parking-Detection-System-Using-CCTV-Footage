# 🅿️ Deep Learning–Based Parking Occupancy Detection from CCTV Footage

> **End-to-end computer vision system** that detects occupied and vacant parking spaces from top-view CCTV camera feeds using state-of-the-art object detection models — with a production-ready real-time inference pipeline.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8%20%7C%20v11-00FFFF?logo=yolo&logoColor=black)](https://docs.ultralytics.com)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)](#license)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Models Trained & Compared](#models-trained--compared)
  - [1 · YOLOv11 (CLI)](#1--yolov11-ultralytics-cli)
  - [2 · YOLOv8 (Python API)](#2--yolov8-ultralytics-python-api)
  - [3 · RF-DETR (Transformer)](#3--rf-detr-roboflow-detection-transformer)
- [Performance Comparison](#performance-comparison)
- [Real-Time Inference Pipeline](#real-time-inference-pipeline)
  - [How It Works](#how-it-works)
  - [Slot Configuration](#slot-configuration)
  - [Usage Examples](#usage-examples)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [License](#license)

---

## Problem Statement

Urban parking is a **$100B+ global market challenge**. Drivers spend an average of **17 minutes per trip** searching for parking, contributing to traffic congestion, fuel waste, and CO₂ emissions. Traditional solutions rely on expensive per-slot hardware sensors (ultrasonic, magnetic, infrared) that cost **$200–$500 per space** to install and maintain.

**This project takes a fundamentally different approach**: leverage existing CCTV infrastructure — already deployed in most parking facilities — and apply deep learning–based object detection to determine slot-level occupancy in real time, at a fraction of the cost.

---

## Solution Overview

| Aspect | Detail |
|---|---|
| **Input** | Top-view CCTV footage (static camera, RTSP stream, or video file) |
| **Detection** | Fine-tuned YOLOv8, YOLOv11, and RF-DETR models classify each vehicle bounding box |
| **Occupancy Logic** | Detection centroids are matched to user-defined slot polygons via ray-casting |
| **Temporal Smoothing** | Hysteresis-based state machine prevents flickering from transient occlusions |
| **Output** | Live annotated overlay, console status, or saved video with per-slot occupancy |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐              │
│  │  CCTV /  │    │    Frame     │    │   Object      │              │
│  │  RTSP /  │───►│   Sampler    │───►│   Detector    │              │
│  │  Webcam  │    │  (N ms/frm)  │    │  (YOLO/DETR)  │              │
│  └──────────┘    └──────────────┘    └───────┬───────┘              │
│                                              │                      │
│                                     Bounding Boxes                  │
│                                     (x1,y1,x2,y2,conf)             │
│                                              │                      │
│                                              ▼                      │
│                                    ┌─────────────────┐              │
│  ┌──────────────┐                  │  Slot Matcher   │              │
│  │  Slot Config │─────────────────►│  (point-in-     │              │
│  │  (JSON)      │  polygon defs    │   polygon)      │              │
│  └──────────────┘                  └────────┬────────┘              │
│                                             │                       │
│                                    candidate states                  │
│                                             │                       │
│                                             ▼                       │
│                                    ┌─────────────────┐              │
│                                    │   Hysteresis    │              │
│                                    │  State Machine  │              │
│                                    │ (occ/emp thresh)│              │
│                                    └────────┬────────┘              │
│                                             │                       │
│                              ┌──────────────┼──────────────┐        │
│                              ▼              ▼              ▼        │
│                        ┌──────────┐  ┌───────────┐  ┌──────────┐   │
│                        │ Annotated│  │  Console  │  │  Saved   │   │
│                        │ Overlay  │  │  Output   │  │  Video   │   │
│                        └──────────┘  └───────────┘  └──────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Models Trained & Compared

All three models were fine-tuned on the **PKLot** dataset (see [Dataset](#dataset)) using Google Colab with GPU acceleration. Training notebooks with full reproducible code are included.

---

### 1 · YOLOv11 (Ultralytics CLI)

> **Notebook**: [`1_YOLOv11_CLI.ipynb`](./1_YOLOv11_CLI.ipynb)

| Parameter | Value |
|---|---|
| Base Model | `yolo11s.pt` (small variant) |
| Training Method | Ultralytics CLI (`yolo task=detect mode=train`) |
| Input Resolution | 640 × 640 |
| Epochs | 10 |
| GPU | NVIDIA T4 (Google Colab) |

**Key Results**:

| Metric | Value |
|---|---|
| **Precision (P)** | 0.97 |
| **Recall (R)** | 0.96 |
| **mAP@0.50** | **0.98** |
| **mAP@0.50:0.95** | **0.80** |

**What this notebook demonstrates**:
- CLI-based model training workflow (no Python API code required)
- Confusion matrix, training loss curves, and validation batch predictions
- Inference on held-out test images with annotated bounding boxes

---

### 2 · YOLOv8 (Ultralytics Python API)

> **Notebook**: [`2_YOLOv8_API.ipynb`](./2_YOLOv8_API.ipynb)

| Parameter | Value |
|---|---|
| Base Model | `yolov8s.pt` (small variant) |
| Training Method | Ultralytics Python API (`model.train()`) |
| Input Resolution | 640 × 640 |
| Epochs | 10 |
| GPU | NVIDIA T4 (Google Colab) |

**Key Results**:

| Metric | Value |
|---|---|
| **Precision (P)** | 0.96 |
| **Recall (R)** | 0.96 |
| **mAP@0.50** | **0.98** |
| **mAP@0.50:0.95** | **0.79** |

**What this notebook demonstrates**:
- Full Python API training workflow (`model.train()`, `model.val()`, `model.predict()`)
- Extended diagnostic visualizations: F1 curve, Precision curve, Recall curve, PR curve, normalized confusion matrix, label correlogram
- Programmatic inference and annotated prediction display
- Deployment-ready code structure for integration into downstream applications

---

### 3 · RF-DETR (Roboflow Detection Transformer)

> **Notebook**: [`3_RF_DETR_Transformer.ipynb`](./3_RF_DETR_Transformer.ipynb)

| Parameter | Value |
|---|---|
| Base Model | RF-DETR Base (DINOv2 backbone) |
| Training Method | `rfdetr` Python API with custom YOLO→COCO label conversion |
| Input Resolution | 640 × 640 |
| Epochs | 3 |
| Batch Size | 8 (with 2× gradient accumulation → effective batch 16) |
| Learning Rate | 1e-4 (with cosine warmup) |
| Mixed Precision | FP16 enabled |
| GPU | NVIDIA A100 (Google Colab) |

**Key Results**:

| Metric | Value |
|---|---|
| **Validation Loss** | **4.6 → 2.6** (43% reduction in 3 epochs) |
| **Inference Threshold** | 0.50 |

**What this notebook demonstrates**:
- **Custom dataset conversion**: YOLO-format labels → COCO-format JSON (annotation parser written from scratch)
- **Transformer-based detection**: DINOv2 vision backbone with deformable attention
- **Advanced training techniques**: mixed-precision (FP16), gradient accumulation, cosine LR warmup, EMA weight smoothing
- Evaluation via `supervision` library: confusion matrix, mAP plot, annotated detection grid (3×3)
- Model checkpoint saving (full model + `model_module.pth` for deployment)

---

## Performance Comparison

| Model | Architecture | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | Training Epochs | GPU Used |
|---|---|---|---|---|---|---|---|
| **YOLOv11s** | CNN (CSPDarknet) | **0.98** | **0.80** | 0.97 | 0.96 | 10 | T4 |
| **YOLOv8s** | CNN (CSPDarknet) | **0.98** | 0.79 | 0.96 | 0.96 | 10 | T4 |
| **RF-DETR Base** | Transformer (DINOv2) | — | — | — | — | 3 | A100 |

> **Key Insight**: Both YOLO variants achieve near-identical performance (~98% mAP@0.50) on the PKLot dataset, suggesting the task is well-suited for single-stage detectors at this resolution. The RF-DETR transformer model shows strong convergence (43% loss reduction) in just 3 epochs, indicating excellent transfer learning from the DINOv2 backbone — a promising direction for more complex parking scenarios with heavy occlusion.

---

## Real-Time Inference Pipeline

> **Script**: [`realtime_inference_pipeline.py`](./realtime_inference_pipeline.py)

A **production-grade Python pipeline** that connects a trained model to a live video feed and outputs per-slot occupancy status in real time.

### How It Works

1. **Frame Sampling** — Reads frames from any OpenCV-compatible source (webcam, RTSP IP camera, video file) at a configurable interval (default: 500ms) to reduce GPU load without sacrificing responsiveness.

2. **Object Detection** — Each sampled frame is passed through a pluggable detector backend:
   - `ultralytics` — for YOLOv8 / YOLOv11 `.pt` weights
   - `torch_script` — for exported TorchScript models (RF-DETR, custom architectures)

3. **Slot Matching (Point-in-Polygon)** — Detection bounding-box centroids are tested against user-defined slot polygons using a ray-casting algorithm. If any detection centroid falls inside a slot polygon, the slot is a candidate for "occupied."

4. **Temporal Hysteresis** — A state machine prevents flickering:
   - A slot transitions to **occupied** only after N consecutive occupied frames (default: 3)
   - A slot transitions to **empty** only after N consecutive empty frames (default: 3)
   - This suppresses noise from pedestrians walking through, brief occlusions, or lighting changes

5. **Output** — Three output modes:
   - **Live overlay**: colour-coded polygons (green = free, red = occupied) with real-time count
   - **Console**: per-frame status with available/occupied counts and detection count
   - **Saved video**: annotated MP4 for review or demo purposes

### Slot Configuration

Define parking slots as polygons in a JSON file. Each slot needs a unique ID and a list of `[x, y]` vertex coordinates matching the camera's perspective:

```json
{
  "slots": [
    {
      "id": "A1",
      "polygon": [[100, 200], [300, 200], [300, 400], [100, 400]]
    },
    {
      "id": "A2",
      "polygon": [[320, 200], [520, 200], [520, 400], [320, 400]]
    },
    {
      "id": "B1",
      "polygon": [[100, 420], [300, 420], [300, 620], [100, 620]]
    }
  ]
}
```

> **Tip**: Use a tool like [LabelMe](https://github.com/labelmeai/labelme) or [CVAT](https://github.com/cvat-ai/cvat) to draw polygon annotations on a reference frame from your camera, then export the coordinates.

### Usage Examples

```bash
# ─── Local webcam (live overlay window) ───
python realtime_inference_pipeline.py \
    --source 0 \
    --slots parking_slots.json \
    --backend ultralytics \
    --weights runs/detect/train_exp/weights/best.pt \
    --display

# ─── RTSP IP camera feed (headless, console output) ───
python realtime_inference_pipeline.py \
    --source rtsp://admin:password@192.168.1.64:554/stream1 \
    --slots parking_slots.json \
    --backend ultralytics \
    --weights best.pt \
    --conf 0.30

# ─── Pre-recorded video → annotated output ───
python realtime_inference_pipeline.py \
    --source parking_footage.mp4 \
    --slots parking_slots.json \
    --backend ultralytics \
    --weights best.pt \
    --save output_annotated.mp4

# ─── Adjust sensitivity for noisy environments ───
python realtime_inference_pipeline.py \
    --source 0 \
    --slots parking_slots.json \
    --weights best.pt \
    --conf 0.35 \
    --occ-thresh 5 \
    --emp-thresh 5 \
    --sample-ms 1000 \
    --display
```

**CLI Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Video source: camera index, video file path, or RTSP URL |
| `--slots` | *(required)* | Path to the parking-slot JSON configuration |
| `--backend` | `ultralytics` | Detection backend: `ultralytics` or `torch_script` |
| `--weights` | `None` | Path to model weights (`.pt` for YOLO, `.ts` for TorchScript) |
| `--imgsz` | `640` | Model input resolution (square) |
| `--conf` | `0.25` | Confidence threshold for detections |
| `--sample-ms` | `500` | Process one frame every N milliseconds |
| `--occ-thresh` | `3` | Consecutive frames to confirm OCCUPIED |
| `--emp-thresh` | `3` | Consecutive frames to confirm EMPTY |
| `--display` | `False` | Show a live annotated overlay window |
| `--save` | `None` | Save annotated video to this path |

### Connecting to a Real CCTV / IP Camera

Most modern IP cameras support **RTSP** (Real Time Streaming Protocol). To integrate:

1. **Find your camera's RTSP URL** — typically in the format:
   ```
   rtsp://<username>:<password>@<camera-ip>:<port>/stream1
   ```
   Common defaults: port `554`, path `/stream1` or `/h264/ch1/main/av_stream`

2. **Run the pipeline** with the RTSP URL as `--source`:
   ```bash
   python realtime_inference_pipeline.py \
       --source "rtsp://admin:admin123@192.168.1.100:554/stream1" \
       --slots my_parking_lot.json \
       --weights best.pt
   ```

3. **For production deployments**, consider:
   - Running on an edge device (NVIDIA Jetson, Intel NUC) co-located with the camera
   - Using a message queue (Redis, MQTT) to publish occupancy updates to a dashboard
   - Wrapping the pipeline in a REST API (FastAPI/Flask) for integration with mobile apps
   - Adding multi-camera support by running parallel pipeline instances

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | [PKLot](http://www.inf.ufpr.br/lesoliv/PKLot/) (Parking Lot Dataset) |
| **Source** | Federal University of Paraná (UFPR), Brazil |
| **Content** | Top-view images of parking lots under varying weather/lighting conditions |
| **Classes** | 2 — `space-empty`, `space-occupied` |
| **Format** | YOLO (used directly for YOLOv8/v11); converted to COCO for RF-DETR |
| **Resolution** | 640 × 640 (pre-resized) |
| **Splits** | Train / Validation / Test |

---

## Project Structure

```
Deep-Learning-Based-Parking-Detection-System-Using-CCTV-Footage/
│
├── 1_YOLOv11_CLI.ipynb               # YOLOv11 training via Ultralytics CLI
├── 2_YOLOv8_API.ipynb                 # YOLOv8 training via Python API (with extended diagnostics)
├── 3_RF_DETR_Transformer.ipynb        # RF-DETR transformer fine-tuning (YOLO→COCO conversion)
├── realtime_inference_pipeline.py     # Production real-time occupancy detection pipeline
├── 4_Project_Report.pdf               # Detailed project report (PDF)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (for training; CPU-only inference is supported but slower)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/dhshah25/Deep-Learning-Based-Parking-Detection-System-Using-CCTV-Footage.git
cd Deep-Learning-Based-Parking-Detection-System-Using-CCTV-Footage

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Training Notebooks

Open any notebook in **Google Colab** (recommended for free GPU access) or locally via Jupyter:

```bash
jupyter notebook 1_YOLOv11_CLI.ipynb
```

### Run the Real-Time Pipeline

```bash
python realtime_inference_pipeline.py \
    --source 0 \
    --slots parking_slots.json \
    --weights path/to/best.pt \
    --display
```

---

## Technical Skills Demonstrated

| Category | Skills |
|---|---|
| **Deep Learning** | Transfer learning, fine-tuning pre-trained models, loss function analysis, hyperparameter tuning |
| **Computer Vision** | Object detection (single-stage & transformer-based), bounding-box regression, NMS, IoU metrics |
| **Model Architectures** | YOLOv8, YOLOv11 (CNN/CSPDarknet), RF-DETR (DINOv2 transformer backbone + deformable attention) |
| **Training Techniques** | Mixed-precision training (FP16), gradient accumulation, cosine LR warmup, EMA weight smoothing |
| **Data Engineering** | YOLO ↔ COCO format conversion, dataset splitting, annotation parsing |
| **Evaluation & Metrics** | mAP@0.50, mAP@0.50:0.95, Precision, Recall, F1-score, confusion matrix, PR curves |
| **Systems / Pipeline** | Real-time video processing (OpenCV), RTSP stream ingestion, temporal hysteresis state machine |
| **Software Engineering** | Type-annotated Python, modular OOP design, CLI argument parsing, pluggable backend pattern |
| **Tools & Frameworks** | PyTorch, Ultralytics, Roboflow, Supervision, OpenCV, Google Colab, Matplotlib |

---

## Limitations & Future Work

| Limitation | Potential Solution |
|---|---|
| Top-view camera only | Train on angled/multi-view datasets for perspective robustness |
| Single-class (vehicle) | Extend to motorcycle, bicycle, EV-charging, handicapped-only detection |
| Manual slot polygon definition | Automated slot detection using line/edge detection or segmentation |
| No dashboard / API | Build a web dashboard (FastAPI + WebSocket) with real-time occupancy map |
| Single-camera only | Multi-camera stitching with homography transforms |
| No cloud deployment | Package with Docker, deploy to AWS/GCP with auto-scaling |

---

## References

1. [PKLot Dataset](http://www.inf.ufpr.br/lesoliv/PKLot/) — Federal University of Paraná
2. [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
3. [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)
4. [RF-DETR — Roboflow](https://github.com/roboflow/rfdetr)
5. [Supervision Library](https://github.com/roboflow/supervision)
6. [DINOv2 — Meta AI](https://arxiv.org/abs/2304.07193)
7. [Albumentations — Data Augmentation](https://arxiv.org/abs/1809.06839)
8. [YOLOv8 — IEEE Paper](https://ieeexplore.ieee.org/document/10533619/)

---

## 📄 Detailed Project Report

A comprehensive PDF report covering objectives, model architecture deep-dives, evaluation metrics, performance comparisons, and future work:

📥 **[Download Project Report (PDF)](./4_Project_Report.pdf)**

---

## License

> ⚠️ **All rights reserved.** This repository is for portfolio and learning reference only. Do **not** reproduce, modify, or redistribute any part of this work without explicit written permission.

---

<p align="center">
  <b>Built by <a href="https://github.com/dhshah25">Dhaval Shah</a></b>
</p>
