#!/bin/bash
# ==============================================================================
# Deploy to Remote Raspberry Pi Over Network (SSH / Tailscale / LAN)
# ==============================================================================
# Usage:
#   ./deploy_to_pi.sh pi@192.168.1.100
#   ./deploy_to_pi.sh pi-vpn
# ==============================================================================

set -e

PI_TARGET="${1:-pi-vpn}"
REMOTE_DIR="~/quantization_edge_ai"

echo "=========================================================="
echo "  Deploying Quantization Edge AI to Raspberry Pi: $PI_TARGET"
echo "=========================================================="

# Create remote directory
ssh "$PI_TARGET" "mkdir -p $REMOTE_DIR"

# Rsync essential deployment assets (excluding bulky caches)
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '*.pth' \
    --exclude '*.pt' \
    --exclude 'esp_quantize/firmware/build' \
    "$(dirname "$0")/../" "$PI_TARGET:$REMOTE_DIR/"

echo "=========================================================="
echo "  Deployment complete! To run on the Pi:"
echo "    ssh $PI_TARGET"
echo "    cd $REMOTE_DIR/pi_quantize"
echo "    ./setup_pi.sh"
echo "    python benchmark_pi.py --model ../quantized_model_int8.onnx"
echo "=========================================================="
