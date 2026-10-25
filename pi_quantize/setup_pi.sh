#!/bin/bash
# ==============================================================================
# Raspberry Pi Edge Deployment Setup Script
# ==============================================================================
# Prepares Debian / Raspberry Pi OS 64-bit for ultra-low latency ONNX Runtime
# execution with ARM NEON SIMD vectorization.
# ==============================================================================

set -e

echo "=========================================================="
echo "  Raspberry Pi Edge AI Deployment Setup"
echo "=========================================================="

# 1. System Dependencies
echo "[1/4] Installing system packages..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    cpufrequtils

# 2. Virtual Environment Setup
echo "[2/4] Setting up Python virtual environment..."
VENV_DIR="$HOME/edge_ai_venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# 3. Python Edge Inference Dependencies
echo "[3/4] Installing ONNX Runtime and scientific stack..."
pip install onnxruntime numpy pillow matplotlib

# 4. Optional CPU Governor Performance Tuning
echo "[4/4] Setting CPU governor to 'performance'..."
if command -v cpufreq-set &> /dev/null; then
    for cpu in $(ls -d /sys/devices/system/cpu/cpu[0-9]* 2>/dev/null); do
        cpu_num=$(basename "$cpu" | tr -dc '0-9')
        sudo cpufreq-set -c "$cpu_num" -g performance 2>/dev/null || true
    done
    echo "CPU governor successfully set to performance mode."
else
    echo "Note: cpufreq-set not found, skipping governor tuning."
fi

echo "=========================================================="
echo "  Setup Complete!"
echo "  To activate your environment:"
echo "    source $VENV_DIR/bin/activate"
echo "  To run benchmark:"
echo "    python benchmark_pi.py --model ../quantized_model_int8.onnx"
echo "=========================================================="
