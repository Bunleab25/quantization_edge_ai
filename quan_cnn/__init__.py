"""
quan_cnn: Neural Network Quantization Toolkit for AdvancedMNISTCNN
===================================================================
Provides Post-Training Quantization (PTQ), module fusion (Conv-BN-ReLU),
calibration, layer sensitivity audits (SQNR, Cosine Similarity),
and edge runtime export (TorchScript, ONNX).
"""

from .prepare import QuantizableAdvancedMNISTCNN, prepare_model_for_quantization
from .calibrate import get_calibration_loader, calibrate_model, convert_to_quantized
from .audit import compute_sqnr, compute_cosine_similarity, run_sensitivity_audit
from .evaluate import evaluate_accuracy_and_latency
from .export import export_quantized_model

__all__ = [
    "QuantizableAdvancedMNISTCNN",
    "prepare_model_for_quantization",
    "get_calibration_loader",
    "calibrate_model",
    "convert_to_quantized",
    "compute_sqnr",
    "compute_cosine_similarity",
    "run_sensitivity_audit",
    "evaluate_accuracy_and_latency",
    "export_quantized_model",
]
