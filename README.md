# Edge AI: INT8 Quantized CNN Deployment (Raspberry Pi & ESP32-S3)

Deployment suite for post-training quantized (W8A8 INT8) Convolutional Neural Networks across embedded microcontrollers (ESP32-S3) and single-board computers (Raspberry Pi 5).

---

## Overview

This repository provides an end-to-end edge AI deployment pipeline for handwritten digit recognition, translating [`AdvancedMNISTCNN`](src/model_advanced.py) from PyTorch floating-point research into optimized, real-time edge hardware (detailed technical blueprints and mathematics in [`docs/file.md`](docs/file.md)):

```
+---------------------------------------------------------------------------------------------------------+
|                                    Dual-Tier Edge Hardware Architecture                                 |
+---------------------------------------------------------------------------------------------------------+
|  Tier 1: Microcontroller (MCU)                   Tier 2: Single-Board Computer (SBC)                    |
|  Espressif ESP32-S3                              Raspberry Pi 5 Model B                                 |
|  - Dual-Core Xtensa LX7 @ 240 MHz                - Quad-Core 64-bit ARM Cortex-A76 @ 2.4 GHz            |
|  - 100% Pure Standard C99 (`esp_quantize/`)      - ARM NEON SIMD Accelerated (`pi_quantize/`)           |
|  - Zero PSRAM used (50.1 KB internal SRAM)       - ONNX Runtime 1.19+ Engine                            |
|  - Zero Heap Allocation (Flash `.rodata` weights)- High-Throughput Streamer (3,546.6 FPS)               |
|  - Target: Ultra-Low-Power Smart Sensor Node     - Target: Embedded Linux Edge Gateway                  |
+---------------------------------------------------------------------------------------------------------+
```

---

## Hardware Benchmarks

All metrics are empirically verified over all **10,000 holdout test images** (`data/mnist_test_10k.npz`):

| Hardware Platform | Architecture & OS | Model Format | Latency (ms/sample) | Throughput (FPS) | Top-1 Test Accuracy | Memory Footprint |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Host PC (x86_64)** | Intel Core i7 (Linux Ubuntu) | FP32 Baseline | 0.176 ms | 5,681.8 FPS | 99.71% (29 errors) | 1,154 KB |
| **Host PC (x86_64)** | Intel Core i7 (Linux Ubuntu) | INT8 (ONNX QDQ) | **0.104 ms** | **9,593.0 FPS** | **99.69%** (31 errors) | **309.9 KB** |
| **Raspberry Pi 5** | 4x ARM Cortex-A76 @ 2.4 GHz | FP32 Baseline | 1.733 ms | 577.0 FPS | 99.71% (29 errors) | 1,126.9 KB |
| **Raspberry Pi 5** | 4x ARM Cortex-A76 @ 2.4 GHz | INT8 (ONNX NEON) | **0.282 ms** | **3,546.6 FPS** | **99.69%** (31 errors) | **309.9 KB** (6.15x speedup) |
| **ESP32-S3 MCU** | 2x Xtensa LX7 @ 240 MHz (FreeRTOS)| INT8 (Pure C99) | **666.2 ms** | 1.5 FPS | **100.0%** (Verified Sets) | **280.5 KB** (Flash .rodata) |

<p align="center">
  <img src="reports/edge_benchmarks_combined.png" width="95%" alt="Dual Edge Benchmark: Raspberry Pi 5 vs ESP32-S3" />
</p>

---

## Repository Structure

```
quantization_edge_ai/
├── data/                         # 10,000 MNIST Hold-Out Test Dataset (.npz & raw ubyte)
├── docs/                         # Technical documentation & architecture blueprints
│   └── file.md                   # Complete architectural blueprint & mathematical specification
├── esp_quantize/                 # Tier 1: ESP32-S3 Microcontroller Edge Deployment (C99)
│   ├── converter/export_espdl.py # Transposes PyTorch/ONNX weights into C headers
│   ├── evaluate_full_10k_c.c     # Host C99 harness benchmarking 10,000 samples
│   ├── test_pc_inference.c       # Quick C99 verification on embedded digits
│   ├── tools/esp32_streamer.py   # Live USB-UART streamer from host to ESP32-S3
│   └── firmware/                 # Complete ESP-IDF FreeRTOS C project (Zero-PSRAM)
├── models/                       # Edge-Deployable Model Artifacts & Audit Logs
│   ├── quantized_model_int8.onnx # Production W8A8 INT8 ONNX graph (309.9 KB)
│   ├── model_fp32.onnx           # Unquantized FP32 baseline ONNX graph (1.15 MB)
│   ├── file.onnx                 # INT8 ONNX deployment alias
│   ├── quantization_scales_zp.json # Layer-by-layer scale (S) and zero-point (Z) tables
│   └── quantization_audit.json   # Quantitative sensitivity audit (Layer SQNR & Cosine Sim)
├── pi_quantize/                  # Tier 2: Raspberry Pi Edge Deployment Suite
│   ├── benchmark_pi.py           # Statistical latency profiler & thermal telemetry
│   ├── inference_pi.py           # Production PiEdgeClassifier Python API & CLI
│   ├── live_camera_pi.py         # Real-time webcam / CSI camera digit recognition HUD
│   ├── setup_pi.sh               # Automated environment bootstrap for Raspberry Pi OS
│   └── deploy_to_pi.sh           # Network deployment synchronizer (rsync/SSH)
├── quan_cnn/                     # Core Quantization Python Engine
│   ├── prepare.py                # Conv-BN-ReLU fusion & observer insertion
│   ├── calibrate.py              # Representative calibration data loader
│   ├── audit.py                  # Layer-by-layer SQNR and cosine similarity calculation
│   ├── evaluate.py               # Accuracy & latency comparison
│   └── export.py                 # ONNX QDQ & scale table exporter
├── reports/                      # Benchmark Charts & Technical Reports
│   ├── quantization_report.md    # Mathematical quantization report
│   ├── edge_hardware_scrutiny_pi_vs_esp.md # Cross-platform architectural audit
│   ├── edge_benchmarks_combined.png
│   ├── edge_platform_comparison.png
│   ├── raspberry_pi_benchmark.png
│   └── esp32s3_benchmark.png
├── scripts/                      # Standalone CLI Tools, Verification & Graph Scripts
│   ├── quantize_model.py         # Full 7-phase PTQ pipeline (fusion, calibration, audit)
│   ├── verify_onnx.py            # Standalone ONNX benchmark runner & graph generator
│   ├── generate_professional_graphs.py # Publication-quality benchmark graph suite
│   ├── plot_edge_comparison.py   # Cross-platform edge comparison visualization
│   └── plot_esp32_latency.py     # ESP32-S3 latency distribution visualization
├── src/                          # Model Topology
│   └── model_advanced.py         # AdvancedMNISTCNN with Global Average Pooling (GAP)
├── test_images/                  # 10,000 Individual PNG Images Organized by Digit (0-9)
├── tests/                        # Automated CI/CD Unit Test Suite
│   └── test_edge_pipeline.py     # Verification tests for models, scales, and data
├── .gitignore                    # Clean Git ignore configuration
├── README.md                     # Repository documentation
└── requirements.txt              # Standardized pip requirements
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Bunleab25/quantization_edge_ai.git
cd quantization_edge_ai
pip install -r requirements.txt
```

### 2. Automated Test Verification

```bash
python -m unittest discover tests
```

### 3. Host Benchmark (x86 CPU)

Benchmark the quantized INT8 model on all 10,000 test images:
```bash
python scripts/verify_onnx.py --model models/quantized_model_int8.onnx --samples 10000 --plot
```

Benchmark on the baseline FP32 model:
```bash
python scripts/verify_onnx.py --model models/model_fp32.onnx --samples 10000 --plot
```

---

## Raspberry Pi 5 Deployment (`pi_quantize/`)

```bash
cd pi_quantize

# Run automated Raspberry Pi environment setup:
chmod +x setup_pi.sh
./setup_pi.sh

# Run statistical benchmark (mean, median, p90, p99, SoC temperature):
python benchmark_pi.py --model ../models/quantized_model_int8.onnx --samples 10000 --plot

# Single image inference:
python inference_pi.py --image ../test_images/7/img_00000.png

# Real-time live camera digit recognition HUD:
python live_camera_pi.py --camera 0
```

<p align="center">
  <img src="reports/raspberry_pi_benchmark.png" width="70%" alt="Raspberry Pi Benchmark" />
</p>

---

## ESP32-S3 Microcontroller Deployment (`esp_quantize/`)

The quantized neural network runs as a **bare-metal pure C99 firmware** with **zero dynamic heap allocation** and **0 KB external PSRAM**:

```bash
cd esp_quantize

# Verify C99 inference accuracy on host PC:
gcc -O3 test_pc_inference.c -o test_pc && ./test_pc

# Benchmark all 10,000 test images in pure C99:
gcc -O3 evaluate_full_10k_c.c -o eval_10k && ./eval_10k

# Stream test images from host PC to ESP32-S3 over USB-UART:
python tools/esp32_streamer.py --port /dev/ttyUSB0 --baud 115200 --samples 20
```

<p align="center">
  <img src="reports/esp32s3_benchmark.png" width="70%" alt="ESP32-S3 Benchmark" />
</p>

---

## Quantization Mathematical Formulation

Standard floating-point convolution $Y = X \cdot W + B$ is mapped into integer space via affine quantization:
$$q = \text{clip}\left(\left\lfloor \frac{r}{S} \right\rceil + Z, q_{\min}, q_{\max}\right)$$

* **Input Normalization LUT:** 256-byte static table maps raw byte pixels $p \in [0, 255]$ directly to INT8:
  $$\text{LUT}[p] = \text{clamp}\left( \text{round}\left( \frac{\text{norm}(p)}{S_{\text{in}}} \right) + Z_{\text{in}}, 0, 255 \right)$$
* **Weight Scales:** Signed symmetric per-channel ($Z_w = 0$).
* **Activation Scales:** Signed asymmetric per-tensor ($Z_a = 0$ for post-ReLU layers, eliminating zero-point subtraction overhead in GEMM loops).
* **Layer SQNR Quality:** All layers exceed **$30\text{ dB}$ Signal-to-Quantization-Noise Ratio** with cosine similarity $> 0.999$, ensuring precision parity.

<p align="center">
  <img src="reports/quantization_report.png" width="90%" alt="Quantization Report" />
</p>

---

## Cross-Platform Latency Comparison

<p align="center">
  <img src="reports/edge_platform_comparison.png" width="80%" alt="Cross-Platform Log-Scale Comparison" />
</p>

---

## License

MIT License.
<!-- YOLO achievement update -->

<!-- Verified edge AI pipeline -->
