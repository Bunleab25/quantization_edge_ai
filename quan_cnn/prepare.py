"""
Model Preparation Module for Post-Training Quantization (PTQ)
=============================================================
Handles:
1. Conv-BN-ReLU fusion (BatchNorm folding into Conv weights).
2. Quantization stubs (QuantStub, DeQuantStub) insertion.
3. Backend configuration (x86, fbgemm, or qnnpack).
4. Insertion of activation observers.
"""

from typing import Tuple, List, Optional
import copy
import torch
import torch.nn as nn
import torch.ao.quantization as aoq

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_advanced import AdvancedMNISTCNN


class QuantizableAdvancedMNISTCNN(nn.Module):
    """
    Quantization-ready wrapper around AdvancedMNISTCNN.
    Inserts QuantStub at model input and DeQuantStub at model output.
    """

    def __init__(self, base_model: Optional[AdvancedMNISTCNN] = None) -> None:
        super().__init__()
        self.quant = aoq.QuantStub()
        self.model = base_model if base_model is not None else AdvancedMNISTCNN()
        self.dequant = aoq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def fuse_model_modules(model: QuantizableAdvancedMNISTCNN) -> QuantizableAdvancedMNISTCNN:
    """
    Fuses Conv2d + BatchNorm2d + ReLU into ConvReLU2d across all 3 stages.
    
    Formula for fused weights:
        W_fused = (gamma / sqrt(var + eps)) * W
        b_fused = (gamma / sqrt(var + eps)) * (b - mean) + beta
    
    Returns the fused model in eval mode.
    """
    model.eval()

    # Fuse Stage 1
    aoq.fuse_modules(
        model.model.stage1,
        [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']],
        inplace=True
    )
    # Fuse Stage 2
    aoq.fuse_modules(
        model.model.stage2,
        [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']],
        inplace=True
    )
    # Fuse Stage 3
    aoq.fuse_modules(
        model.model.stage3,
        [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']],
        inplace=True
    )

    return model


def prepare_model_for_quantization(
    fp32_model: AdvancedMNISTCNN,
    backend: str = "x86",
) -> QuantizableAdvancedMNISTCNN:
    """
    Prepares an FP32 AdvancedMNISTCNN for static PTQ:
    1. Wraps inside QuantizableAdvancedMNISTCNN.
    2. Fuses Conv-BN-ReLU blocks.
    3. Attaches W8A8 qconfig (per-channel symmetric weights, per-tensor asymmetric activations).
    4. Calls torch.ao.quantization.prepare() to insert observers.
    
    Args:
        fp32_model: Pretrained FP32 AdvancedMNISTCNN instance.
        backend: Quantization engine ('x86', 'fbgemm', or 'qnnpack').
        
    Returns:
        Prepared model with active observers ready for calibration.
    """
    # Clone to preserve original FP32 model untouched
    base_copy = copy.deepcopy(fp32_model)
    base_copy.eval()

    # Wrap in quantizable container
    qmodel = QuantizableAdvancedMNISTCNN(base_model=base_copy)

    # 1. Fuse modules
    qmodel = fuse_model_modules(qmodel)

    # 2. Select backend & qconfig
    supported = torch.backends.quantized.supported_engines
    if backend not in supported:
        fallback = "x86" if "x86" in supported else supported[0]
        print(f"[Prepare] Backend '{backend}' not supported, falling back to '{fallback}'")
        backend = fallback

    torch.backends.quantized.engine = backend
    qmodel.qconfig = aoq.get_default_qconfig(backend)

    # 3. Insert observers
    aoq.prepare(qmodel, inplace=True)
    print(f"[Prepare] Observers attached using backend: '{backend}' (W8A8 per-channel symmetric weights)")
    return qmodel
