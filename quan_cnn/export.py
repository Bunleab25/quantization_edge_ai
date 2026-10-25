"""
Export Module for Quantized Models
==================================
Exports quantized models to multiple deployment targets:
1. PyTorch Quantized State Dict (.pth)
2. TorchScript Traced Binary (.pt) for C++ / PyTorch Mobile / Edge deployment
3. Parameter Scales & Zero-Points Table (.json) for edge integer kernels
4. ONNX QDQ format (.onnx) if onnx runtime is installed
"""

from typing import Dict, Any, Tuple
import os
import json
import torch
import torch.nn as nn


def extract_quantization_parameters(quant_model: nn.Module) -> Dict[str, Any]:
    """
    Extracts numerical scale (s) and zero-point (z) for each quantized layer.
    Essential for auditing and deploying to integer hardware kernels.
    """
    params: Dict[str, Any] = {}

    for name, module in quant_model.named_modules():
        # Check for quantized conv / linear / stub modules
        if hasattr(module, "scale") and hasattr(module, "zero_point"):
            s = module.scale
            z = module.zero_point
            params[name] = {
                "type": module.__class__.__name__,
                "activation_scale": float(s.item() if isinstance(s, torch.Tensor) else s),
                "activation_zero_point": int(z.item() if isinstance(z, torch.Tensor) else z),
            }

        # Check for quantized weight attributes
        if hasattr(module, "weight"):
            w = module.weight() if callable(module.weight) else module.weight
            if hasattr(w, "qscheme"):
                scheme_str = str(w.qscheme())
                if "per_channel" in scheme_str:
                    w_scales = w.q_per_channel_scales().tolist()
                    w_zps = w.q_per_channel_zero_points().tolist()
                    if name not in params:
                        params[name] = {"type": module.__class__.__name__}
                    params[name]["weight_qscheme"] = "per_channel_symmetric"
                    params[name]["num_channels"] = len(w_scales)
                    params[name]["weight_scale_sample"] = [round(x, 6) for x in w_scales[:5]]
                else:
                    if name not in params:
                        params[name] = {"type": module.__class__.__name__}
                    params[name]["weight_qscheme"] = "per_tensor"
                    params[name]["weight_scale"] = float(w.q_scale())
                    params[name]["weight_zero_point"] = int(w.q_zero_point())

    return params


def export_quantized_model(
    quant_model: nn.Module,
    output_dir: str = "checkpoints",
    dummy_shape: Tuple[int, ...] = (1, 1, 28, 28),
) -> Dict[str, str]:
    """
    Exports the quantized model to deployable formats.
    
    Returns:
        Dict of exported file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}

    dummy_input = torch.randn(*dummy_shape)

    # 1. PyTorch Quantized State Dict
    pth_path = os.path.join(output_dir, "quantized_model_int8.pth")
    torch.save(quant_model.state_dict(), pth_path)
    exported_files["state_dict"] = pth_path
    pth_size = os.path.getsize(pth_path) / 1024.0
    print(f"[Export] Saved PyTorch state_dict: {pth_path} ({pth_size:.1f} KB)")

    # 2. TorchScript Traced Binary (Mobile / C++ / Embedded runtime)
    pt_path = os.path.join(output_dir, "quantized_model_int8.pt")
    try:
        scripted = torch.jit.trace(quant_model, dummy_input)
        torch.jit.save(scripted, pt_path)
        exported_files["torchscript"] = pt_path
        pt_size = os.path.getsize(pt_path) / 1024.0
        print(f"[Export] Saved TorchScript binary: {pt_path} ({pt_size:.1f} KB)")
    except Exception as e:
        print(f"[Export] TorchScript tracing warning: {e}")

    # 3. Scales & Zero-Points Table
    json_path = os.path.join(output_dir, "quantization_scales_zp.json")
    scales_zp = extract_quantization_parameters(quant_model)
    with open(json_path, "w") as f:
        json.dump(scales_zp, f, indent=2)
    exported_files["scales_zp"] = json_path
    print(f"[Export] Saved Quantization Parameters JSON: {json_path}")

    # 4. ONNX QDQ Export (Optional)
    onnx_path = os.path.join(output_dir, "quantized_model_int8.onnx")
    try:
        import onnx
        torch.onnx.export(
            quant_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=13,
            dynamo=False,
        )
        exported_files["onnx"] = onnx_path
        onnx_size = os.path.getsize(onnx_path) / 1024.0
        print(f"[Export] Saved ONNX QDQ model: {onnx_path} ({onnx_size:.1f} KB)")
    except ImportError:
        print("[Export] Note: 'onnx' package not installed. TorchScript (.pt) is ready for C++/edge deployment.")
    except Exception as e:
        print(f"[Export] ONNX export note: {e}")

    return exported_files
