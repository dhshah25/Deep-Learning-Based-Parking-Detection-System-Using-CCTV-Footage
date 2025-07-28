# 🚗 Smart Parking Detection System (YOLOv11, YOLOv8, RF-DETR)

This project implements a deep learning–based solution for detecting parking occupancy using top-view CCTV footage. We explore and compare three object detection approaches: **YOLOv11 (Ultralytics CLI)**, **YOLOv8 (Python API)**, and **RF-DETR (transformer-based model)**. All models are trained and evaluated on the **PKLot** dataset.

> ⚠️ **All rights reserved.** This repository is for portfolio and learning reference only. Do **not** reproduce, modify, or redistribute any part of this work without explicit permission.

---

## Problem Statement

Finding vacant parking spots in crowded areas is inefficient and frustrating. Traditional sensor-based solutions are costly and complex to maintain. This system may be used to detect occupied and vacant parking spaces in real-time from CCTV footage using deep learning.

---

## Models Implemented

### 1. YOLOv11 (Ultralytics CLI)

- Trained via CLI on PKLot dataset (640x640 format)
- Achieved **98% AP@0.50** and **80% mAP@[.50:.95]**
- CLI commands used: `train`, `val`, `predict`
- Outputs: `results.png`, `val_batch0_pred.jpg`, prediction samples

📁 **Notebook**: `1_YOLOv11_CLI.ipynb`

---

### 2. YOLOv8 (Ultralytics Python API)

- Trained via Python API using `model.train()` and `model.predict()`
- Custom visualization and annotated prediction overlays
- Future deployment-ready for integration into a smart parking app

📁 **Notebook**: `2_YOLOv8_API.ipynb`

---

### 3. RF-DETR (Roboflow Detection Transformer)

- Converted YOLO labels → COCO format using a custom parser
- Fine-tuned transformer-based RF-DETR with:
  - DINOv2 backbone
  - Mixed-precision training (fp16)
  - Cosine LR warmup
  - Gradient accumulation
  - EMA smoothing
- Validation loss reduced from **4.6 → 2.6**
- Outputs include confusion matrix, precision-recall curve, and annotated detection grid

📁 **Notebook**: `3_RF_DETR_Transformer.ipynb`

---

## 📄 Detailed Report

Includes:
- Objective
- Model architecture explanations
- Evaluation metrics
- Performance comparisons
- Limitations and future work

📄 **File**: `4_Project_Report.pdf`

---

## Setup & Requirements

To install dependencies:

```bash
pip install -r requirements.txt
