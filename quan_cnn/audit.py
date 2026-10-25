"""
Sensitivity Audit Module for Quantized Neural Networks
======================================================
Implements quantitative verification of quantization noise:
1. Signal-to-Quantization-Noise Ratio (SQNR): 10 * log10(||x_fp32||^2 / ||x_fp32 - x_hat||^2)
2. Cosine Similarity: (x_fp32 . x_hat) / (||x_fp32|| * ||x_hat||)
3. Layer-by-layer audit table with pass/investigate threshold checks.
"""

from typing import Dict, Any, List, Tuple
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_sqnr(fp32_tensor: torch.Tensor, quant_tensor: torch.Tensor) -> float:
    """
    Computes Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB).
    Formula: 10 * log10( sum(x_fp32^2) / sum((x_fp32 - x_hat)^2) )
    
    A higher SQNR indicates lower quantization error.
    Heuristic guideline for INT8: flag if SQNR < 30 dB.
    """
    if hasattr(quant_tensor, "dequantize"):
        quant_tensor = quant_tensor.dequantize()

    x_f = fp32_tensor.detach().float()
    x_q = quant_tensor.detach().float()

    signal_pwr = torch.sum(x_f ** 2)
    noise_pwr = torch.sum((x_f - x_q) ** 2)

    eps = 1e-10
    if noise_pwr.item() < eps:
        return 100.0  # Perfect reconstruction

    sqnr = 10.0 * torch.log10((signal_pwr + eps) / (noise_pwr + eps)).item()
    return float(sqnr)


def compute_cosine_similarity(fp32_tensor: torch.Tensor, quant_tensor: torch.Tensor) -> float:
    """
    Computes Cosine Similarity between FP32 activations and dequantized INT8 activations.
    Formula: (x_fp32 . x_hat) / (||x_fp32|| * ||x_hat||)
    
    Guideline for INT8: flag if cosine < 0.99.
    """
    if hasattr(quant_tensor, "dequantize"):
        quant_tensor = quant_tensor.dequantize()

    x_f = fp32_tensor.detach().float().contiguous().reshape(-1)
    x_q = quant_tensor.detach().float().contiguous().reshape(-1)

    norm_f = torch.norm(x_f)
    norm_q = torch.norm(x_q)

    eps = 1e-10
    if norm_f.item() < eps and norm_q.item() < eps:
        return 1.0

    cosine = (torch.dot(x_f, x_q) / (norm_f * norm_q + eps)).item()
    return float(cosine)


def run_sensitivity_audit(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    output_path: str = "checkpoints/quantization_audit.json",
) -> Dict[str, Any]:
    """
    Extracts intermediate layer activations for both models on the same batch,
    computes SQNR and Cosine similarity per layer, flags degraded layers,
    and returns a structured audit report.
    """
    fp32_model.eval().to(device)
    quant_model.eval().to(device)

    # Layers to hook
    # Map from friendly name to (fp32_submodule, quant_submodule)
    layer_map = {
        "Stage 1 (Dual 32-ch)": (fp32_model.stage1, quant_model.model.stage1),
        "Stage 2 (Dual 64-ch)": (fp32_model.stage2, quant_model.model.stage2),
        "Stage 3 (Dual 128-ch)": (fp32_model.stage3, quant_model.model.stage3),
        "Global Avg Pool (GAP)": (fp32_model.gap, quant_model.model.gap),
        "Linear Classifier": (fp32_model.classifier, quant_model.model.classifier),
    }

    fp32_activations: Dict[str, torch.Tensor] = {}
    quant_activations: Dict[str, torch.Tensor] = {}
    hooks: List[Any] = []

    def make_hook(store: Dict[str, torch.Tensor], name: str):
        def hook_fn(mod, inp, out):
            if hasattr(out, "dequantize"):
                store[name] = out.dequantize().detach().cpu()
            else:
                store[name] = out.detach().cpu()
        return hook_fn

    for name, (fp_mod, q_mod) in layer_map.items():
        hooks.append(fp_mod.register_forward_hook(make_hook(fp32_activations, name)))
        hooks.append(q_mod.register_forward_hook(make_hook(quant_activations, name)))

    # Run single evaluation batch
    sample_images, _ = next(iter(data_loader))
    sample_images = sample_images.to(device)

    with torch.no_grad():
        fp32_model(sample_images)
        quant_model(sample_images)

    # Clean up hooks immediately
    for h in hooks:
        h.remove()

    # Compute audit metrics
    audit_results: Dict[str, Dict[str, Any]] = {}
    print("\n" + "=" * 80)
    print("                 🔍 LAYER-BY-LAYER QUANTIZATION SENSITIVITY AUDIT")
    print("=" * 80)
    print(f"{'Layer / Block':<26} | {'SQNR (dB)':<11} | {'Cosine Sim':<11} | {'Status':<14} | {'Action'}")
    print("-" * 80)

    for name in layer_map.keys():
        act_f = fp32_activations[name]
        act_q = quant_activations[name]

        sqnr = compute_sqnr(act_f, act_q)
        cos = compute_cosine_similarity(act_f, act_q)

        # Flag thresholds from quant-engineer skill
        sqnr_pass = sqnr >= 30.0
        cos_pass = cos >= 0.99
        is_pass = sqnr_pass and cos_pass

        status = "✅ PASS" if is_pass else "⚠️ FLAG"
        action = "Keep INT8" if is_pass else "Investigate / Mixed-Precision"

        audit_results[name] = {
            "sqnr_db": round(sqnr, 2),
            "cosine_similarity": round(cos, 5),
            "status": "PASS" if is_pass else "FLAG",
            "decision": action,
        }

        print(f"{name:<26} | {sqnr:>7.2f} dB   | {cos:>9.5f}   | {status:<14} | {action}")

    print("=" * 80)

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"[Audit] Audit report saved to: {output_path}")

    return audit_results
