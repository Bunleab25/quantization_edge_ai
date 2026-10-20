# quantization_edge_ai

Edge-deployable **INT8 Quantized CNN** and **FP32 baseline** models in ONNX format, paired with the full **10,000 MNIST test dataset** for benchmark verification, edge AI deployment, and embedded hardware accelerators (e.g., ESP32, STM32, ARM Cortex-M, Raspberry Pi, ONNX Runtime).

---

## 📦 Repository Contents

| File / Folder | Format / Size | Description |
| :--- | :--- | :--- |
| **`quantized_model_int8.onnx`** | ONNX (317 KB) | Post-Training Quantized (PTQ W8A8) INT8 model for edge deployment |
| **`model_fp32.onnx`** | ONNX (1.15 MB) | Full FP32 floating-point baseline model |
| **`quantization_scales_zp.json`** | JSON (2.3 KB) | Layer-by-layer scale ($S$) and zero-point ($Z$) quantization parameters |
| **`quantization_audit.json`** | JSON (695 B) | Layer SQNR (dB) and Cosine Similarity sensitivity verification metrics |
| **`verify_onnx.py`** | Python | Standalone benchmark script measuring latency & accuracy |
| **`data/mnist_test_10k.npz`** | NumPy (1.6 MB) | 10,000 test images `(10000, 28, 28)` and labels `(10000,)` |
| **`data/MNIST/raw/`** | MNIST ubyte | Standard MNIST test dataset (`t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` & `.gz`) |
| **`test_images/`** | PNG Images (2.6 MB) | 10,000 individual 28x28 grayscale PNG images organized by digit (`0/` to `9/`) |

---

## ⚡ Quantization & Performance Benchmark

| Metric | FP32 Baseline | INT8 Quantized | Delta / Speedup |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **99.35%** | **99.28%** | **-0.07% drop** (Budget $\le$ 0.3%) |
| **Model Size** | **1,154 KB** | **317 KB** | **3.64x Compression** |
| **Inference Latency** | ~0.42 ms / image | **~0.11 ms / image** | **~3.8x Faster on CPU** |
| **Target Precision** | FP32 | INT8 (W8A8 Static PTQ) | Integer GEMM accelerated |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install onnxruntime numpy pillow
```

### 2. Run Verification Benchmark

Benchmark the quantized INT8 model on all 10,000 test images:

```bash
python verify_onnx.py --model quantized_model_int8.onnx --samples 10000
```

Benchmark on the FP32 baseline model:

```bash
python verify_onnx.py --model model_fp32.onnx --samples 10000
```

### 3. Load Model in Python (ONNX Runtime)

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
<!-- Pair Extraordinaire collaboration -->
