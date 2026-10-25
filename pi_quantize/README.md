# Raspberry Pi Edge Deployment: INT8 Quantized CNN (`pi_quantize`)

High-performance, edge-accelerated inference of [`AdvancedMNISTCNN`](../src/model_advanced.py) on **Raspberry Pi 5 / 4 / CM4** Single-Board Computers using **ONNX Runtime** and **ARM NEON SIMD vectorization**.

---

## Table of Contents
1. [Hardware Specifications & ARM Acceleration](#1-hardware-specifications--arm-acceleration)
2. [Empirical Benchmark Results](#2-empirical-benchmark-results)
3. [Directory Architecture](#3-directory-architecture)
4. [Quick Start & Setup](#4-quick-start--setup)
5. [Benchmarking on 10,000 Test Images](#5-benchmarking-on-10000-test-images)
6. [Single Image Inference & Python API](#6-single-image-inference--python-api)
7. [Live Camera Digit Recognition (HUD)](#7-live-camera-digit-recognition-hud)
8. [Thermal & CPU Governor Tuning](#8-thermal--cpu-governor-tuning)

---

## 1. Hardware Specifications & ARM Acceleration

| Component | Specification |
| :--- | :--- |
| **Edge Hardware** | **Raspberry Pi 5 Model B** |
| **SoC / Architecture** | Broadcom BCM2712 — Quad-core 64-bit ARM Cortex-A76 @ 2.4 GHz |
| **SIMD Vector Engine** | ARM NEON 128-bit vector execution pipeline |
| **Operating System** | Raspberry Pi OS (Debian 12 Bookworm, 64-bit Kernel) |
| **Inference Engine** | ONNX Runtime 1.19+ (CPUExecutionProvider with INT8 GEMM) |
| **RAM** | 8 GB LPDDR4X SDRAM |

### Why ARM NEON Accelerates INT8:
Floating-point arithmetic (FP32) executes 32-bit operations sequentially per vector lane. With Post-Training Quantization (W8A8), four 8-bit integers are packed into a single 32-bit register, allowing ARM NEON to execute **$4\times$ more Multiply-Accumulate (MAC) operations per clock cycle**, cutting memory bus bandwidth by $73\%$.

---

## 2. Empirical Benchmark Results

Evaluated across the **full 10,000 MNIST holdout test set** (`data/mnist_test_10k.npz`):

| Performance Metric | FP32 Baseline (`model_fp32.onnx`) | INT8 Quantized (`quantized_model_int8.onnx`) | Edge Impact / Speedup |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **99.71%** (9,971 / 10,000) | **99.69%** (9,969 / 10,000) | **-0.02% drop** (only 2 errors diff) |
| **Average Latency** | 1.733 ms / sample | **0.282 ms / sample** | ⚡ **6.15x Faster on ARM CPU** |
| **Throughput (FPS)** | 577.0 samples / s | **3,546.6 samples / s** | 🚀 **+515% Throughput Gain** |
| **Total Evaluation Time**| 17.332 s | **2.820 s** | **14.51 s Saved** |
| **Model Size** | 1,126.9 KB (1.15 MB) | **309.9 KB** | 📦 **3.64x Storage & RAM Reduction** |

<p align="center">
  <img src="../reports/raspberry_pi_benchmark.png" width="70%" alt="Raspberry Pi Benchmark" />
</p>

---

## 3. Directory Architecture

```
pi_quantize/
├── README.md               # Complete Raspberry Pi documentation & benchmarks
├── benchmark_pi.py         # Full 10,000-image benchmark with latency stats & telemetry
├── inference_pi.py         # Production inference class (PiEdgeClassifier) & CLI tool
├── live_camera_pi.py       # Real-time webcam / CSI camera digit HUD classifier
├── setup_pi.sh             # Automated setup script for Raspberry Pi OS
├── deploy_to_pi.sh         # Network sync script (rsync over SSH/VPN)
└── __init__.py             # Python module initialization
```

---

## 4. Quick Start & Setup

### Automatic Setup (Raspberry Pi OS)
```bash
cd pi_quantize
chmod +x setup_pi.sh deploy_to_pi.sh
./setup_pi.sh
```

### Manual Installation
```bash
pip install onnxruntime numpy pillow matplotlib opencv-python
```

---

## 5. Benchmarking on 10,000 Test Images

To benchmark the INT8 quantized model with latency distribution statistics (`mean`, `median`, `p90`, `p99`, `min`, `max`, `std`), CPU temperature telemetry, and evaluation graph generation:

```bash
python benchmark_pi.py --model ../models/quantized_model_int8.onnx --samples 10000 --plot
```

To compare against the floating-point baseline:
```bash
python benchmark_pi.py --model ../models/model_fp32.onnx --samples 10000 --plot
```

---

## 6. Single Image Inference & Python API

### CLI Prediction:
```bash
python inference_pi.py --image ../test_images/7/img_00000.png
```

### Python API Usage:
```python
from pi_quantize.inference_pi import PiEdgeClassifier

# Initialize session
classifier = PiEdgeClassifier()

# Run prediction
digit, confidence, probs, latency_ms = classifier.predict("test_digit.png")
print(f"Predicted Digit: {digit} ({confidence:.2f}%) in {latency_ms:.3f} ms")
```

---

## 7. Live Camera Digit Recognition (HUD)

Point any USB webcam or Raspberry Pi Camera Module v2 / v3 at handwritten digits:

```bash
python live_camera_pi.py --camera 0
```
- Real-time contour isolation & center-of-mass centering ($28 \times 28$).
- Live Heads-Up Display (HUD) showing predicted digit, confidence percentage, and inference FPS.

---

## 8. Thermal & CPU Governor Tuning

To guarantee jitter-free, deterministic real-time inference without thermal throttling:

```bash
# Set CPU governor to performance across all 4 cores:
sudo cpufreq-set -c 0 -g performance
sudo cpufreq-set -c 1 -g performance
sudo cpufreq-set -c 2 -g performance
sudo cpufreq-set -c 3 -g performance

# Check CPU temperature:
vcgencmd measure_temp
```
