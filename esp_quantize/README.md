# ESP32-S3 Edge Deployment: Advanced MNIST CNN (Pure C Inference Engine)

High-performance, bare-metal edge AI deployment of [`AdvancedMNISTCNN`](../../src/model_advanced.py) ($99.70\%$ INT8 test accuracy) onto an **Espressif ESP32-S3 microcontroller** using **100% pure standard C99** firmware.

---

## Table of Contents
1. [Hardware Overview & Pinout](#1-hardware-overview--pinout)
2. [Directory Architecture](#2-directory-architecture)
3. [Quantization & Math Formulation](#3-quantization--math-formulation)
4. [Memory Architecture & Zero-PSRAM Design](#4-memory-architecture--zero-psram-design)
5. [Step-by-Step Build & Flash Guide](#5-step-by-step-build--flash-guide)
6. [Dual-Mode Verification & Empirical Results](#6-dual-mode-verification--empirical-results)
   - [A. Automatic Boot Self-Test (10 Digits)](#a-automatic-boot-self-test-10-digits)
   - [B. Live USB-Serial PC Streaming](#b-live-usb-serial-pc-streaming)
7. [Host-Side C Test Harness](#7-host-side-c-test-harness)
8. [Debugging & Root Cause Case Study](#8-debugging--root-cause-case-study)

---

## 1. Hardware Overview & Pinout

The firmware interfaces directly with a standard **HD44780 16×2 Character LCD** driven by a **PCF8574 I2C Backpack**:

```
+-------------------------------------------------------------+
|                        ESP32-S3 SoC                         |
|                                                             |
|   [GPIO 1]  ---------------------> PCF8574 I2C SDA          |
|   [GPIO 2]  ---------------------> PCF8574 I2C SCL          |
|   [5V/VBUS] ---------------------> PCF8574 VCC (5V Power)   |
|   [GND]     ---------------------> PCF8574 GND (Ground)     |
|                                                             |
|   [GPIO 43] ---------------------> UART0 TXD (USB Serial)   |
|   [GPIO 44] ---------------------> UART0 RXD (USB Serial)   |
+-------------------------------------------------------------+
```

| ESP32-S3 Pin | Peripheral Pin | Function | Configuration File |
| :--- | :--- | :--- | :--- |
| **GPIO 1** | **SDA** | I2C Data Line (100 kHz standard mode) | `firmware/main/app_config.h` |
| **GPIO 2** | **SCL** | I2C Clock Line (100 kHz standard mode) | `firmware/main/app_config.h` |
| **5V / VBUS** | **VCC** | 5V Power for LCD backlight & contrast | Hardware header |
| **GND** | **GND** | Common reference ground | Hardware header |
| **GPIO 43/44** | **TX/RX** | USB-UART console and frame streaming (115200 baud) | Default UART0 |

> **Display Note:** If text appears faint on the LCD, adjust the blue trim potentiometer on the back of the PCF8574 backpack until character pixels are sharp.

---

## 2. Directory Architecture

```
esp_quantize/
├── converter/
│   └── export_espdl.py           # Exports PyTorch weights -> model_data.h & test_samples.h
│
├── firmware/                     # Self-contained ESP-IDF v5.5 project
│   ├── CMakeLists.txt            # Root project CMake
│   ├── sdkconfig.defaults        # ESP32-S3 240MHz + SIMD optimization defaults
│   └── main/
│       ├── CMakeLists.txt        # Component CMake with -O3 -funroll-loops
│       ├── app_config.h          # Hardware pinout & model macros
│       ├── i2c_lcd.h / .c        # PCF8574 HD44780 16x2 LCD driver
│       ├── espdl_model.h / .c    # Pure C vectorized inference engine + 256-byte LUT
│       ├── model_data.h          # W8A8 INT8 quantized weights in Flash (.rodata, ~280.5 KB)
│       ├── test_samples.h        # 10 embedded MNIST digits (0..9) for boot self-test
│       └── main.c                # Pure C dual-mode execution (Boot self-test + UART stream task)
│
├── test_pc_inference.c           # Host-side C test harness for rapid PC validation
├── tools/
│   └── esp32_streamer.py         # Host Python script to stream MNIST frames over USB serial
└── README.md                     # Comprehensive deployment engineering guide
```

---

## 3. Quantization & Math Formulation

The inference engine performs integer convolution and fixed-point requantization matching PyTorch W8A8 calibration:

### A. Input Normalization via 256-Byte LUT
Raw camera/UART pixels $p \in [0, 255]$ are prequantized using a 256-byte lookup table:
$$\text{norm}(p) = \frac{\frac{p}{255.0} - 0.1307}{0.3081}$$
$$\text{LUT}[p] = \text{clamp}\left( \text{round}\left( \frac{\text{norm}(p)}{S_{\text{in}}} \right) + Z_{\text{in}}, 0, 255 \right)$$
where $S_{\text{in}} = 0.02554421$ and $Z_{\text{in}} = 17$. Background black pixels ($0$) map directly to integer $0$.

### B. 3×3 Convolution with Per-Channel Requantization
For a feature map with input channels $C_{\text{in}}$, output channels $C_{\text{out}}$, input scale $S_{\text{in}}$, and output scale $S_{\text{out}}$:
1. **Integer Multiply-Accumulate (MAC):**
   $$\text{acc}[y, x, oc] = B_q[oc] + \sum_{dy=-1}^{1} \sum_{dx=-1}^{1} \sum_{ic=0}^{C_{\text{in}}-1} (X[py, px, ic] - Z_{\text{in}}) \cdot W[oc, dy+1, dx+1, ic]$$
2. **Layer-Specific Quantized Bias:**
   $$B_q[oc] = \text{round}\left( \frac{B_{\text{float}}[oc]}{S_{\text{in}} \cdot S_w[oc]} \right)$$
3. **Effective Multiplier per Channel:**
   $$M[oc] = \frac{S_{\text{in}} \cdot S_w[oc]}{S_{\text{out}}}$$
4. **Requantization & Post-ReLU Clamping:**
   $$Y[y, x, oc] = \text{clamp}\left( \text{round}(\text{acc} \cdot M[oc]) + Z_{\text{out}}, Z_{\text{out}}, 255 \right)$$
   For all post-ReLU intermediate layers, $Z_{\text{out}} = 0$ and $Z_{\text{in}} = 0$.

### C. Global Average Pooling (GAP) & Classifier Head
1. **Global Average Pooling:**
   $$\text{GAP}[c] = \frac{1}{49} \sum_{i=0}^{48} X[i, c], \quad c \in [0, 127]$$
2. **Classifier Head (128 -> 10 Logits):**
   $$\text{logit}[c] = \left( \sum_{i=0}^{127} (\text{GAP}[i] - Z_{\text{in}}) \cdot W_{\text{classifier}}[c, i] \right) \cdot (S_{\text{in}} \cdot S_w[c]) + B_{\text{float}}[c]$$
3. **Stable Softmax Probabilities:**
   $$P[c] = \frac{\exp(\text{logit}[c] - \max(\text{logit}))}{\sum_{k=0}^{9} \exp(\text{logit}[k] - \max(\text{logit}))}$$

---

## 4. Memory Architecture & Zero-PSRAM Design

The neural network is architected to execute without requiring external PSRAM:

| Memory Component | Allocation Type | Location | Size | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Model Weights & Biases** | `static const` | Flash (`.rodata`) | **$280.5\text{ KB}$** | Mapped via flash cache; 0 KB heap used |
| **Ping-Pong Buffer A** | `static` global | Internal SRAM | **$25.1\text{ KB}$** | Aligned to 16 bytes for vector memory loads |
| **Ping-Pong Buffer B** | `static` global | Internal SRAM | **$25.1\text{ KB}$** | Aligned to 16 bytes for vector memory loads |
| **Input Buffer** | `static` global | Internal SRAM | **$784\text{ B}$** | 28×28 grayscale input tensor |
| **Normalization LUT** | `static` global | Internal SRAM | **$256\text{ B}$** | Precomputed uint8 transformation table |
| **FreeRTOS Stack** | `TaskStack` | Internal SRAM | **$8\text{ KB}$** | Main inference and UART streamer task |
| **Total Runtime RAM** | - | **Internal SRAM** | **$\approx 59.2\text{ KB}$** | **$0\text{ KB}$ PSRAM required** |

---

## 5. Step-by-Step Build & Flash Guide

### Step 1: Re-export Weights & Test Samples (Host PC)
```bash
conda activate ai-ml
cd ~/research_academic/mnist_cnn_project
python esp_quantize/converter/export_espdl.py
```

### Step 2: Set Up ESP-IDF v5.5 Toolchain
```bash
. $HOME/.espressif/v5.5.5/esp-idf/export.sh
```

### Step 3: Build the Firmware
```bash
cd ~/research_academic/mnist_cnn_project/esp_quantize/firmware

# Set target chip to ESP32-S3 (first run)
idf.py set-target esp32s3

# Build application binary
idf.py build
```

### Step 4: Flash & Monitor
```bash
idf.py -p /dev/ttyUSB0 flash monitor
```
*(To exit the monitor, press `Ctrl + ]`)*

---

## 6. Dual-Mode Verification & Empirical Results

### A. Automatic Boot Self-Test (10 Digits)
Immediately upon power-on, the ESP32-S3 performs a self-test across all 10 embedded MNIST digits:

```text
I (411) APP_MAIN: ============================================================
I (418) APP_MAIN:          🧪 RUNNING BOOT SELF-TEST ON 10 EMBEDDED DIGITS
I (424) APP_MAIN: ============================================================
I (1132) APP_MAIN: Sample 0 | Expected: 0 | Predicted: 0 | Conf:  96.1% | Time: 666.36 ms | ✅ PASS
I (1848) APP_MAIN: Sample 1 | Expected: 1 | Predicted: 1 | Conf:  95.7% | Time: 666.18 ms | ✅ PASS
I (2564) APP_MAIN: Sample 2 | Expected: 2 | Predicted: 2 | Conf:  96.3% | Time: 666.22 ms | ✅ PASS
I (3280) APP_MAIN: Sample 3 | Expected: 3 | Predicted: 3 | Conf:  91.1% | Time: 666.22 ms | ✅ PASS
I (3996) APP_MAIN: Sample 4 | Expected: 4 | Predicted: 4 | Conf:  95.5% | Time: 666.20 ms | ✅ PASS
I (4712) APP_MAIN: Sample 5 | Expected: 5 | Predicted: 5 | Conf:  88.4% | Time: 666.21 ms | ✅ PASS
I (5428) APP_MAIN: Sample 6 | Expected: 6 | Predicted: 6 | Conf:  97.0% | Time: 666.22 ms | ✅ PASS
I (6144) APP_MAIN: Sample 7 | Expected: 7 | Predicted: 7 | Conf:  95.1% | Time: 666.20 ms | ✅ PASS
I (6860) APP_MAIN: Sample 8 | Expected: 8 | Predicted: 8 | Conf:  97.4% | Time: 666.22 ms | ✅ PASS
I (7576) APP_MAIN: Sample 9 | Expected: 9 | Predicted: 9 | Conf:  96.5% | Time: 666.19 ms | ✅ PASS
I (7626) APP_MAIN: ------------------------------------------------------------
I (7626) APP_MAIN: Self-Test Result: 10 / 10 Correct | Avg Latency: 666.22 ms
I (7628) APP_MAIN: ============================================================
```

The I2C 16×2 LCD summarizes the result:
```
Row 0: Test: 10/10 Pass
Row 1: Avg: 666.2 ms
```

### B. Live USB-Serial PC Streaming
After boot self-test, the ESP32-S3 listens on UART0 for 784-byte frames:

```bash
conda activate ai-ml
cd ~/research_academic/mnist_cnn_project
python esp_quantize/tools/esp32_streamer.py --port /dev/ttyUSB0 --samples 5
```

Output:
```text
================================================================================
        🚀 ESP32-S3 SERIAL MNIST STREAMER & BENCHMARK TOOL
================================================================================
Target Port:      /dev/ttyUSB0
Baud Rate:        115200
Test Samples:     5
================================================================================
Connected to /dev/ttyUSB0 successfully!

#    | Expected  | Predicted  | Conf (%)  | On-Chip (ms)  | Roundtrip  | Result
--------------------------------------------------------------------------------
1    | 7         | 7          |   95.1%   |    665.30 ms  |   918.7 ms | ✅ PASS
2    | 2         | 2          |   96.3%   |    665.33 ms  |   919.0 ms | ✅ PASS
3    | 1         | 1          |   95.7%   |    665.28 ms  |   919.1 ms | ✅ PASS
4    | 0         | 0          |   96.1%   |    665.33 ms  |   918.9 ms | ✅ PASS
5    | 4         | 4          |   95.5%   |    665.30 ms  |   919.3 ms | ✅ PASS
================================================================================
                    🏁 STREAMING BENCHMARK RESULTS
================================================================================
Total Sent:               5
Accuracy:                 5/5 (100.0%)
Avg On-Chip Inference:    665.31 ms
Avg Host Round-Trip:      918.98 ms
================================================================================
```

---

## 7. Host-Side C Test Harness

To enable lightning-fast debugging and verify pure C mathematical parity without reflashing the microcontroller, a standalone host C test harness is included:

```bash
cd ~/research_academic/mnist_cnn_project
gcc -O3 -Iesp_quantize/firmware/main esp_quantize/test_pc_inference.c -lm -o esp_quantize/test_pc_bin
./esp_quantize/test_pc_bin
```

Output:
```text
============================================================
   VERIFYING FAST PURE C INFERENCE ENGINE ON PC (HOST)
============================================================
Digit 0 | Expected: 0 | Predicted: 0 | Conf:  96.1% | Time:  3.34 ms | PASS
Digit 1 | Expected: 1 | Predicted: 1 | Conf:  95.7% | Time:  2.92 ms | PASS
Digit 2 | Expected: 2 | Predicted: 2 | Conf:  96.3% | Time:  2.91 ms | PASS
Digit 3 | Expected: 3 | Predicted: 3 | Conf:  91.1% | Time:  2.94 ms | PASS
Digit 4 | Expected: 4 | Predicted: 4 | Conf:  95.5% | Time:  2.90 ms | PASS
Digit 5 | Expected: 5 | Predicted: 5 | Conf:  88.4% | Time:  2.88 ms | PASS
Digit 6 | Expected: 6 | Predicted: 6 | Conf:  97.0% | Time:  2.91 ms | PASS
Digit 7 | Expected: 7 | Predicted: 7 | Conf:  95.1% | Time:  2.89 ms | PASS
Digit 8 | Expected: 8 | Predicted: 8 | Conf:  97.4% | Time:  2.89 ms | PASS
Digit 9 | Expected: 9 | Predicted: 9 | Conf:  96.5% | Time:  2.88 ms | PASS
============================================================
Summary: 10 / 10 Correct!
============================================================
```

---

## 8. Debugging & Root Cause Case Study

During development, two critical edge bugs were isolated and eliminated following the **Debug Mantra**:

### Bug 1: Constant Prediction of Class `2` (Accuracy Bug)
* **Root Cause 1 (Layout Transposition):** PyTorch weights are shaped `(Cout, Cin, 3, 3)`. The C loops assumed channels-last `(Cout, 3, 3, Cin)`. Directly flattening the PyTorch weights transposed the channel and spatial kernel axes, scrambling filter matrices.
* **Root Cause 2 (Dummy Requantization Scale):** A placeholder constant `0.0015f` was used instead of per-channel effective multipliers $M[oc] = \frac{S_{\text{in}} \cdot S_w[oc]}{S_{\text{out}}}$.
* **Root Cause 3 (Bias Scale Inconsistency):** Downstream biases were divided by the QuantStub scale ($0.02554$) instead of each layer's actual input scale.
* **Solution:** In `converter/export_espdl.py`, weights are transposed to `(Cout, 3, 3, Cin)` before serialization, layer-specific quantized biases $B_q$ are calculated with exact input scales, and arrays of `eff_scale[oc]` are compiled into Flash.

### Bug 2: Task Watchdog Timer (TWDT) Timeout (Performance Bug)
* **Root Cause:** 10 consecutive scalar inferences ($29.1\text{ M MACs}$ total) on a 240 MHz core took ~11 seconds in a continuous blocking loop without yielding CPU time to the FreeRTOS IDLE task, triggering the 5-second Watchdog reset.
* **Solution:** Inserted `vTaskDelay(pdMS_TO_TICKS(50))` between test sample evaluations in `firmware/main/main.c`, enabling clean context switches and automatic watchdog resets.
