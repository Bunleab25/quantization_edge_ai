# quantization_edge_ai

Edge-deployable **INT8 Quantized CNN** and **FP32 baseline** models in ONNX and Pure C format, paired with the full **10,000 MNIST test dataset** for benchmark verification, edge AI deployment, and embedded hardware accelerators (e.g., ESP32, STM32, ARM Cortex-M, Raspberry Pi, ONNX Runtime).

---

## 📦 Repository Contents

| File / Folder | Format / Size | Description |
| :--- | :--- | :--- |
| **`quantized_model_int8.onnx`** | ONNX (317 KB) | Post-Training Quantized (PTQ W8A8) INT8 model for edge deployment |
| **`model_fp32.onnx`** | ONNX (1.15 MB) | Full FP32 floating-point baseline model |
| **`quantization_scales_zp.json`** | JSON (2.3 KB) | Layer-by-layer scale ($S$) and zero-point ($Z$) quantization parameters |
| **`quantization_audit.json`** | JSON (695 B) | Layer SQNR (dB) and Cosine Similarity sensitivity verification metrics |
| **`quantize_model.py`** | Python | End-to-end 7-phase PTQ pipeline (fusion, calibration, audit, evaluation, export) |
| **`quan_cnn/`** | Python Package | Complete quantization engine (`prepare`, `calibrate`, `audit`, `evaluate`, `export`) |
| **`src/model_advanced.py`** | PyTorch | `AdvancedMNISTCNN` architecture topology with Global Average Pooling (GAP) |
| **`esp_quantize/`** | Pure C99 Firmware | Zero-PSRAM ESP32-S3 bare-metal C99 inference engine, converter, and UART streamer |
| **`reports/`** | Reports & Plots | Publication-grade quantization reports, audit logs, and hardware benchmark graphs |
| **`verify_onnx.py`** | Python | Standalone benchmark script measuring latency, accuracy, and graph plotting |
| **`plot_edge_comparison.py`** | Python | Cross-platform edge SBC vs. MCU performance visualization |
| **`plot_esp32_latency.py`** | Python | ESP32-S3 latency distribution and per-digit response visualization |
| **`data/mnist_test_10k.npz`** | NumPy (1.6 MB) | 10,000 test images `(10000, 28, 28)` and labels `(10000,)` |
| **`data/MNIST/raw/`** | MNIST ubyte | Standard MNIST test dataset (`t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` & `.gz`) |
| **`test_images/`** | PNG Images (2.6 MB) | 10,000 individual 28x28 grayscale PNG images organized by digit (`0/` to `9/`) |

---

## ⚡ Real Hardware Edge Benchmarks

### 🍓 Real Hardware Benchmark on Raspberry Pi 5 (ARM Cortex-A76 @ 2.4 GHz)

*Empirically measured over all 10,000 hold-out test samples on Raspberry Pi 5 (`quantization_edge_ai` via ONNX Runtime):*

| Metric | FP32 Baseline (`model_fp32.onnx`) | INT8 Quantized (`quantized_model_int8.onnx`) | Hardware Impact / Edge Speedup |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **99.71%** (9,971 / 10,000) | **99.69%** (9,969 / 10,000) | **-0.02% drop** (only 2 errors difference!) |
| **Inference Latency** | 1.733 ms / sample | **0.282 ms / sample** | ⚡ **6.15x Faster on ARM CPU** |
| **Inference Throughput** | 577.0 samples / s | **3,546.6 samples / s** | 🚀 **+515% Throughput Gain** |
| **Total Test Evaluation Time** | 17.332 s | **2.820 s** | **14.51 s Saved** |
| **Model Binary Size** | 1,126.9 KB (1.15 MB) | **309.9 KB** | 📦 **3.64x Storage & RAM Compression** |

### 📟 Microcontroller Benchmark on ESP32-S3 (Dual-Core Xtensa LX7 @ 240 MHz)

*Bare-metal pure C99 inference engine with zero dynamic heap allocation (`malloc/free`) and 0 KB PSRAM:*

| Metric | PC PyTorch (FP32) | Raspberry Pi 5 (ONNX INT8) | ESP32-S3 Microcontroller (Pure C INT8) |
| :--- | :---: | :---: | :---: |
| **Operating System** | Linux (Ubuntu) | Debian 12 (Linux 64-bit) | FreeRTOS (Bare-metal firmware) |
| **Execution Engine** | PyTorch 2.13 (CUDA/CPU) | ONNX Runtime 1.19 (ARM NEON) | Custom Pure C99 Engine (`-O3`) |
| **Weight Footprint** | 1,154 KB | 317 KB (ONNX File) | **280.5 KB** (C Array in Flash `.rodata`) |
| **Runtime SRAM / DRAM**| > 500 MB | > 120 MB | **50.1 KB** (Internal SRAM Ping-Pong Buffers)|
| **External Memory** | System RAM | 8 GB LPDDR4X | **0 KB PSRAM required** |
| **Inference Latency** | 0.18 ms | 0.104 ms | **666.2 ms** |
| **Accuracy on MNIST** | 99.71% | 99.69% | **100.0%** (Verified boot & UART test sets) |

### 💻 Host PC Benchmark (x86 CPU)

| Metric | FP32 Baseline | INT8 Quantized | Delta / Speedup |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | 99.71% | 99.69% (ONNX) / 99.70% (PyTorch) | -0.01% to -0.02% drop |
| **Inference Latency** | ~0.176 ms / sample | ~0.104 ms / sample | ~1.69x Faster on CPU |
| **Throughput** | ~5,681 samples / s | ~9,593 samples / s | Integer GEMM accelerated |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install onnxruntime numpy pillow matplotlib
```

### 2. Run Verification Benchmark

Benchmark the quantized INT8 model on all 10,000 test images:

```bash
python verify_onnx.py --model quantized_model_int8.onnx --samples 10000 --plot
```

Benchmark on the FP32 baseline model:

```bash
python verify_onnx.py --model model_fp32.onnx --samples 10000 --plot
```

### 3. Run Post-Training Quantization Pipeline

To re-run the full 7-phase PTQ workflow, calibration, and sensitivity audit:

```bash
python quantize_model.py
```

### 4. Load Model in Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Initialize inference session
session = ort.InferenceSession("quantized_model_int8.onnx")
input_name = session.get_inputs()[0].name

# Load and preprocess any 28x28 grayscale test image
img = Image.open("test_images/7/img_00000.png").convert("L")
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - 0.1307) / 0.3081  # Normalize
input_tensor = arr.reshape(1, 1, 28, 28)

# Run inference
logits = session.run(None, {input_name: input_tensor})[0]
predicted_digit = int(np.argmax(logits, axis=1)[0])
print(f"Predicted Digit: {predicted_digit}")
```

---

## 🔬 Quantization Parameters & Layer SQNR

Extracted affine quantization mapping:
$$q = \text{clip}\left(\left\lfloor \frac{r}{S} \right\rceil + Z, q_{\min}, q_{\max}\right)$$

Numerical scales ($S$) and zero-points ($Z$) for each operator are tabulated in [`quantization_scales_zp.json`](./quantization_scales_zp.json). All layers exceed the 30 dB Signal-to-Quantization-Noise Ratio (SQNR) threshold, verifying minimal precision loss.

---

## 📄 License

MIT License.
<!-- YOLO achievement update -->

<!-- Verified edge AI pipeline -->
