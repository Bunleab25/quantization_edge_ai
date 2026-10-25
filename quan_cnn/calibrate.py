"""
Calibration and Conversion Module for Post-Training Quantization
================================================================
Passes representative samples through the prepared model to observe
activation min/max ranges, then converts float operations to INT8.
"""

from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as aoq


def get_calibration_loader(
    data_dir: str = "./data",
    num_samples: int = 256,
    batch_size: int = 32,
) -> DataLoader:
    """
    Creates a clean calibration DataLoader from the MNIST dataset.
    
    Calibration data uses standard test-time normalization:
    mean=0.1307, std=0.3081. No random augmentations are applied.
    
    Args:
        data_dir: Directory where MNIST dataset is stored.
        num_samples: Number of representative images (Skill recommendation: 128 - 512).
        batch_size: Mini-batch size for calibration feedforward.
        
    Returns:
        DataLoader providing calibration batches.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    indices = list(range(min(num_samples, len(test_dataset))))
    subset = Subset(test_dataset, indices)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    print(f"[Calibrate] Prepared calibration loader with {len(subset)} samples (batch_size={batch_size})")
    return loader


def calibrate_model(
    prepared_model: torch.nn.Module,
    calibration_loader: DataLoader,
    device: str = "cpu",
) -> None:
    """
    Runs forward passes through the prepared model without gradient calculation.
    Observers capture the dynamic range [min, max] of activations.
    
    Args:
        prepared_model: Model with attached quantization observers.
        calibration_loader: DataLoader supplying representative samples.
        device: Device to run calibration on (typically 'cpu').
    """
    prepared_model.eval()
    prepared_model.to(device)

    print("[Calibrate] Feeding calibration data through observers...")
    batches = 0
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            prepared_model(images)
            batches += 1

    print(f"[Calibrate] Completed calibration over {batches} batches.")


def convert_to_quantized(prepared_model: torch.nn.Module) -> torch.nn.Module:
    """
    Freezes observer scales and zero-points, and swaps floating-point
    modules to quantized INT8 modules.
    
    Returns:
        Converted INT8 model.
    """
    print("[Calibrate] Converting model to INT8 quantized representation...")
    quantized_model = aoq.convert(prepared_model, inplace=False)
    print("[Calibrate] Conversion complete! All weights and activations quantized to INT8.")
    return quantized_model
