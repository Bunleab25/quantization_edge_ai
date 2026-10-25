# 📊 Post-Training Quantization (W8A8 PTQ) — Full Technical Report

**Model:** `AdvancedMNISTCNN` (3-Stage Dual-Conv + Global Average Pooling)  
**Framework:** PyTorch 2.13.0+cu130  
**Dataset:** MNIST Handwritten Digits (10 classes, 28×28 grayscale)  
**Execution Date:** 2026-10-15  
**Author:** Bunleab (Module 4 — CNNs & Overfitting)

---

## 1. What We Started With (FP32 Baseline)

The `AdvancedMNISTCNN` was trained for 12 epochs using CosineAnnealingLR + Label Smoothing (0.05) + augmented data (rotation 12°, affine, perspective distortion). The trained checkpoint was saved as `best_advanced_model.pth`.

| Property | Value |
|:---|:---|
| Architecture | 3 × `DualConvBlock` (Conv-BN-ReLU-Conv-BN-ReLU) + GAP + Linear(128→10) |
| Total Parameters | 252,490 |
| Weight Precision | FP32 (32-bit floating point, 4 bytes per weight) |
| Checkpoint Size | 3,493 KB (3.5 MB — includes optimizer state for Adam) |
| Weights-only Size | ~1,010 KB (252,490 × 4 bytes) |
| Best Validation Accuracy | 99.63% (Epoch 11) |
| Diagnosis | REGULARIZED GOOD FIT (OPTIMAL) |

---

## 2. What We Did (Step-by-Step Quantization Process)

### Step 1: Model Preparation — Conv-BN-ReLU Fusion

**Why:** During training, BatchNorm is a separate operation with its own learnable parameters ($\gamma, \beta$) and running statistics ($\mu, \sigma^2$). At inference time, BN is mathematically equivalent to a linear transform that can be folded directly into the preceding Conv weights. This eliminates 6 entire BN layers and simplifies the graph for quantization.

**The math:**

$$W_{\text{fused}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot W_{\text{conv}}$$

$$b_{\text{fused}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (b_{\text{conv}} - \mu) + \beta$$

**What happened:** All 6 BatchNorm layers across 3 stages were fused into their preceding Conv layers. After fusion, each `DualConvBlock` became `ConvReLU2d → ConvReLU2d → MaxPool2d` (no standalone BN or ReLU modules).

| Stage | Fused Patterns | Result |
|:---|:---|:---|
| Stage 1 | `[conv1, bn1, relu1]` + `[conv2, bn2, relu2]` | 2 × `ConvReLU2d` + `MaxPool2d` |
| Stage 2 | `[conv1, bn1, relu1]` + `[conv2, bn2, relu2]` | 2 × `ConvReLU2d` + `MaxPool2d` |
| Stage 3 | `[conv1, bn1, relu1]` + `[conv2, bn2, relu2]` | 2 × `ConvReLU2d` + `Identity` (no pool) |
| **Total** | **6 fusions performed** | **6 BN layers eliminated** |

### Step 2: QuantStub / DeQuantStub Insertion

A `QuantStub` was inserted at the model input to convert FP32 input tensors to quantized INT8 format, and a `DeQuantStub` at the output to convert the final INT8 logits back to FP32. This boundary defines where the integer arithmetic pipeline starts and ends.

### Step 3: Observer Attachment & Quantization Config

**Quantization scheme chosen:**

| Component | Precision | Range | Granularity | Formula |
|:---|:---|:---|:---|:---|
| **Weights** | INT8 Signed | $[-127, +127]$ | **Per-Channel** (per output feature map) | $s_w = \frac{\max\|w_c\|}{127}$, $z_w = 0$ (symmetric) |
| **Activations** | INT8 Signed | $[-128, +127]$ | **Per-Tensor** | $s_a = \frac{\beta - \alpha}{255}$, $z_a = q_{\min} - \text{round}(\alpha / s_a)$ (asymmetric) |
| **Bias** | INT32 | Full range | Per-Channel | $s_b = s_{\text{input}} \cdot s_{\text{weight}}$, $z_b = 0$ |

**Why per-channel for weights:** Each Conv filter (output channel) has its own magnitude distribution. A single per-tensor scale would be dominated by the filter with the largest values, wasting representable range for all other filters. Per-channel gives each filter its own optimal scale.

**Why asymmetric for activations:** Post-ReLU activations are always $\geq 0$. Symmetric quantization would waste half the INT8 range ($[-128, -1]$) on values that never appear. Asymmetric sets $z \neq 0$ so that the zero-point shifts the entire $[0, \max]$ range to fill $[-128, 127]$ efficiently.

**Backend:** `x86` (Intel FBGEMM vectorized INT8 GEMM kernels). Compatible with `qnnpack` for ARM/mobile deployment.

### Step 4: Calibration (256 Representative Samples)

**What:** Ran 256 unaugmented MNIST test images (8 batches × 32 images) through the model while MinMax observers recorded the empirical $[\min, \max]$ of every activation tensor.

**Why 256 samples:** The quant-engineer runbook recommends 128–512 calibration samples. For MNIST (simple digit images with stable distributions), 256 is more than sufficient to capture representative activation ranges.

**Critical rule followed:** Calibration used the **exact same preprocessing** as inference — `ToTensor() + Normalize(mean=0.1307, std=0.3081)`. No training augmentations (rotation, perspective, affine) were applied — those would distort the true activation ranges and produce incorrect observer statistics.

### Step 5: INT8 Conversion

After calibration, `torch.ao.quantization.convert()` replaced all floating-point modules with their quantized equivalents:

- `Conv2d` → `QuantizedConvReLU2d` (weights stored as INT8 tensors)
- `Linear` → `QuantizedLinear` (weights stored as INT8 tensors)
- Scale ($s$) and zero-point ($z$) frozen per layer

---

## 3. Quantization Parameters — Per-Layer Scale & Zero-Point Table

These are the exact numerical values frozen into the quantized model. Any edge runtime must use these identical values.

### Input Quantization (QuantStub)

| Parameter | Value | Meaning |
|:---|:---|:---|
| Activation Scale ($s$) | **0.02554** | Each INT8 step represents 0.02554 in the original FP32 range |
| Activation Zero-Point ($z$) | **17** | The INT8 value 17 maps to FP32 value 0.0 |

### Convolution Layers (ConvReLU2d — Fused)

| Layer | Act Scale ($s_a$) | Act $z_a$ | Weight Scheme | Channels | Weight Scale Sample (first 5 channels) |
|:---|:---:|:---:|:---|:---:|:---|
| **stage1.conv1** | 0.05181 | 0 | Per-Channel Symmetric | 32 | [0.002925, 0.005922, 0.007880, 0.006986, 0.005302] |
| **stage1.conv2** | 0.04854 | 0 | Per-Channel Symmetric | 32 | [0.000857, 0.001697, 0.000927, 0.000936, 0.001095] |
| **stage2.conv1** | 0.04258 | 0 | Per-Channel Symmetric | 64 | [0.000931, 0.001170, 0.000941, 0.001246, 0.001046] |
| **stage2.conv2** | 0.04316 | 0 | Per-Channel Symmetric | 64 | [0.000726, 0.000682, 0.000917, 0.000694, 0.001538] |
| **stage3.conv1** | 0.03703 | 0 | Per-Channel Symmetric | 128 | [0.001696, 0.001273, 0.000997, 0.001578, 0.000918] |
| **stage3.conv2** | 0.04393 | 0 | Per-Channel Symmetric | 128 | [0.001533, 0.001409, 0.001283, 0.001566, 0.001490] |

> **Note:** Activation zero-point = 0 for all ConvReLU2d layers because ReLU guarantees outputs $\geq 0$, so the observer sees only non-negative values and the optimal zero-point is 0.

### Classifier (Linear)

| Parameter | Value |
|:---|:---|
| Activation Scale ($s_a$) | **0.06059** |
| Activation Zero-Point ($z_a$) | **50** |
| Weight Scheme | Per-Channel Symmetric |
| Output Channels | 10 (one per digit class) |
| Weight Scale Sample | [0.001838, 0.001963, 0.002294, 0.001951, 0.001928] |

> **Note:** Classifier has $z_a = 50$ because the final Linear layer outputs raw class scores (logits), which naturally include both positive and negative numbers.

---

## 4. Layer-by-Layer Sensitivity Audit Results

After quantization, we compared the FP32 and INT8 intermediate activations at each stage to verify signal fidelity.

### Metrics Used

**SQNR (Signal-to-Quantization-Noise Ratio):**

$$\text{SQNR} = 10 \cdot \log_{10}\!\left(\frac{\|x_{\text{fp32}}\|^2}{\|x_{\text{fp32}} - \hat{x}_{\text{int8}}\|^2}\right) \quad \text{[dB]}$$

Measures how much of the original signal power is preserved vs. how much was lost to quantization rounding. Higher is better.

**Cosine Similarity:**

$$\cos\theta = \frac{x_{\text{fp32}} \cdot \hat{x}_{\text{int8}}}{\|x_{\text{fp32}}\| \cdot \|\hat{x}_{\text{int8}}\|}$$

Measures directional alignment between the original and quantized activation vectors. 1.0 = perfect match.

### Audit Thresholds (from quant-engineer runbook)

| Precision | Flag if SQNR Below | Flag if Cosine Below |
|:---|:---|:---|
| **INT8** | **30 dB** | **0.99** |
| INT6 | 22 dB | 0.98 |
| INT4 | 12 dB | 0.95 |

### Audit Results

| Layer / Stage Block | SQNR (dB) | Margin Above Threshold | Cosine Similarity | Status | Decision |
|:---|:---:|:---:|:---:|:---:|:---|
| **Stage 1 (Dual 32-ch)** | **33.37** | +3.37 dB | **0.99979** | ✅ PASS | Keep INT8 |
| **Stage 2 (Dual 64-ch)** | **32.47** | +2.47 dB | **0.99972** | ✅ PASS | Keep INT8 |
| **Stage 3 (Dual 128-ch)** | **33.75** | +3.75 dB | **0.99980** | ✅ PASS | Keep INT8 |
| **Global Avg Pool (GAP)** | **31.78** | +1.78 dB | **0.99968** | ✅ PASS | Keep INT8 |
| **Linear Classifier** | **34.75** | +4.75 dB | **0.99985** | ✅ PASS | Keep INT8 |

**Conclusion:** All 5 audit points exceeded the 30 dB SQNR bar by +1.78 to +4.75 dB, and maintained $\geq 0.99968$ cosine similarity. No outlier channels, no degraded feature maps, and no need for mixed precision or QAT.

---

## 5. Final Verification Benchmark (FP32 vs INT8)

Evaluated on the complete MNIST hold-out test set (10,000 images) with identical preprocessing:

| Metric | FP32 Baseline | INT8 Quantized | Delta / Impact | Pass/Fail |
|:---|:---:|:---:|:---:|:---:|
| **Top-1 Test Accuracy** | **99.71%** | **99.70%** | **−0.01%** drop | ✅ PASS ($\leq$ 0.30%) |
| **Total Test Errors** | 29 / 10,000 | 30 / 10,000 | **+1 misclassification** | Near-zero degradation |
| **Model Size (State Dict)** | ~1,010 KB | **304.5 KB** | **3.31× smaller** | ✅ ~254.5 KB raw weights (304.5 KB with file header) |
| **TorchScript Binary** | N/A | **345.7 KB** | Standalone executable | ✅ No Python required |
| **ONNX QDQ Model** | N/A | **309.9 KB** | Cross-platform portable | ✅ Validated with ORT |
| **CPU Latency (per image)** | 0.504 ms | **0.272 ms** | **1.85× faster** | ✅ Real-time capable |
| **CPU Latency (per batch of 128)** | 63.84 ms | **34.47 ms** | **1.85× faster** | ✅ High throughput |
| **Inference Throughput** | ~1,984 img/s | ~3,676 img/s | **+85% throughput** | ✅ |

---

## 6. Model Size Breakdown — Where Did the Bytes Go?

### FP32 (Before Quantization)

| Component | # Params | × Bytes Each | = Size |
|:---|:---:|:---:|:---:|
| Conv weights (6 layers) | 251,200 | × 4 (FP32) | 1,004.8 KB |
| Linear weights + bias | 1,290 | × 4 (FP32) | 5.2 KB |
| **Total weights** | **252,490** | | **~1,010 KB** |

### INT8 (After Quantization)

| Component | # Params | × Bytes Each | = Size |
|:---|:---:|:---:|:---:|
| Conv weights (6 layers) | 251,200 | × 1 (INT8) | 251.2 KB |
| Linear weights + bias | 1,290 | × 1 (INT8) | 1.3 KB |
| Scale & Zero-Point metadata | 7 layers | × ~16 bytes | 0.1 KB |
| Bias (INT32) | ~480 | × 4 (INT32) | 1.9 KB |
| Serialization overhead | — | — | ~50 KB |
| **Total (state dict)** | | | **~304.5 KB** |
| **Total (TorchScript .pt)** | | | **~345.7 KB** |
| **Total (ONNX .onnx)** | | | **~309.9 KB** |

**Compression ratio:** 1,010 KB ÷ 304.5 KB = **3.31×** (state dict) or 1,010 KB ÷ 345.7 KB = **2.92×** (TorchScript)

---

## 7. Exported Deployment Artifacts

All deployment-ready files are in `checkpoints/`:

| # | File | Size | Format | Target Runtimes |
|:---:|:---|:---:|:---|:---|
| 1 | `quantized_model_int8.onnx` | **309.9 KB** | ONNX QDQ (QuantizeLinear / DequantizeLinear) | Cross-platform: PC (x86), Raspberry Pi 5 (ARM), Mobile, Web (WASM) |
| 2 | `quantized_model_int8.pt` | **345.7 KB** | TorchScript Binary | C++, PyTorch Mobile, Embedded (see ARM note below) |
| 3 | `quantized_model_int8.pth` | **304.5 KB** | PyTorch State Dict | Python (PyTorch quantized model reload) |
| 4 | `quantization_scales_zp.json` | **2.3 KB** | JSON Lookup Table | Bare-metal integer kernels, custom FPGA/ASIC inference |
| 5 | `quantization_audit.json` | **0.7 KB** | JSON Audit Log | Documentation / reproducibility |
| 6 | `quantization_report.png` | **~350 KB** | PNG 3-Panel Chart | Visual verification for presentations / reports |

![Quantization Verification Report](quantization_report.png)


> 💡 **Edge Deployment Note (x86 PC vs. Raspberry Pi ARM):**
> - The currently generated `quantized_model_int8.pt` was quantized using `--backend x86` (optimized for Intel/AMD PC CPUs).
> - On the **Raspberry Pi** (ARM Cortex CPU), the ONNX QDQ model (`quantized_model_int8.onnx`) was verified over all 10,000 MNIST test samples via ONNX Runtime:

### 🍓 Empirical Raspberry Pi Hardware Benchmark (Full 10,000 Test Set)

| Metric | FP32 Baseline (`model_fp32.onnx`) | INT8 Quantized (`quantized_model_int8.onnx`) | Hardware Impact / Edge Speedup |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **99.71%** (9,971 / 10,000) | **99.69%** (9,969 / 10,000) | **-0.02% drop** (only 2 errors difference!) |
| **Inference Latency** | 1.733 ms / sample | **0.282 ms / sample** | ⚡ **6.15x Faster on ARM CPU** |
| **Inference Throughput** | 577.0 samples / s | **3,546.6 samples / s** | 🚀 **+515% Throughput Gain** |
| **Total Test Evaluation Time** | 17.332 s | **2.820 s** | **14.51 s Saved** |
| **Model Binary Size** | 1,126.9 KB (1.15 MB) | **309.9 KB** | 📦 **3.64x Storage & RAM Compression** |

---

## 8. Quantization Pipeline Code Structure

```
research_academic/mnist_cnn_project/
├── quantize_model.py                # End-to-end CLI: python quantize_model.py [--calib_samples 256]
│
├── quan_cnn/                        # Modular quantization package
│   ├── __init__.py                  # Public API exports
│   ├── prepare.py                   # Phase 1: Conv-BN-ReLU fusion + observer attachment
│   ├── calibrate.py                 # Phase 2: Calibration data loading + INT8 conversion
│   ├── audit.py                     # Phase 3: SQNR + Cosine similarity computation
│   ├── evaluate.py                  # Phase 4: Accuracy + latency benchmarking
│   └── export.py                    # Phase 5: TorchScript, ONNX, state_dict, scales/zp export
│
├── checkpoints/
│   ├── best_advanced_model.pth      # Original FP32 model (3,493 KB — with optimizer)
│   ├── quantized_model_int8.onnx    # ONNX QDQ export (309.9 KB)
│   ├── quantized_model_int8.pt      # TorchScript binary (345.7 KB)
│   ├── quantized_model_int8.pth     # PyTorch quantized state dict (304.5 KB)
│   ├── quantization_scales_zp.json  # Scale & zero-point per layer
│   └── quantization_audit.json      # Layer SQNR & cosine log
│
└── reports/
    ├── quantization_report.md       # This document
    └── quantization_report.png      # 3-panel visual verification chart
```

---

## 9. How to Reproduce

```bash
# Activate environment
conda activate ai-ml

# Navigate to project
cd ~/research_academic/mnist_cnn_project

# Option 1: Standard PC inference (Intel / AMD x86 CPUs)
python quantize_model.py --calib_samples 256 --backend x86

# Option 2: Raspberry Pi 5 / Mobile inference (ARM CPUs via QNNPACK)
python quantize_model.py --calib_samples 256 --backend qnnpack
```

---

## 10. Summary

| What | Before (FP32) | After (INT8 PTQ) | Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | 99.71% | 99.70% | −0.01% (negligible) |
| **Model Size** | 1,010 KB | 304.5 KB | **3.31× smaller** |
| **Inference Speed** | 0.504 ms/img | 0.272 ms/img | **1.85× faster** |
| **Deployable Formats** | Python `.pth` only | ONNX + TorchScript + `.pth` | **3 edge formats** |
| **Layers Flagged** | N/A | 0 / 5 | **All PASS** |
| **QAT Required?** | N/A | **No** | PTQ was sufficient |
