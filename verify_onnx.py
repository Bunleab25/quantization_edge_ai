#!/usr/bin/env python3
"""
ONNX Model Verification & Benchmarking Script
=============================================
Evaluates FP32 and INT8 ONNX models on the 10,000 MNIST test images.
Measures Top-1 test accuracy and inference latency per sample.

Requirements:
    pip install onnxruntime numpy pillow

Usage:
    python verify_onnx.py
    python verify_onnx.py --model quantized_model_int8.onnx --samples 1000
    python verify_onnx.py --model model_fp32.onnx --samples 10000
"""

import argparse
import os
import sys
import time
import numpy as np
import onnxruntime as ort

def load_test_dataset(data_dir: str = "data"):
    """Loads 10,000 MNIST test images and labels from npz or raw ubyte."""
    npz_path = os.path.join(data_dir, "mnist_test_10k.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data["images"], data["labels"]

    raw_img = os.path.join(data_dir, "MNIST", "raw", "t10k-images-idx3-ubyte")
    raw_lbl = os.path.join(data_dir, "MNIST", "raw", "t10k-labels-idx1-ubyte")
    if os.path.exists(raw_img) and os.path.exists(raw_lbl):
        with open(raw_img, "rb") as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with open(raw_lbl, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return images, labels

    raise FileNotFoundError(f"Test dataset not found in {data_dir}!")

def preprocess_mnist(images: np.ndarray) -> np.ndarray:
    """Standard MNIST normalization: (x / 255.0 - 0.1307) / 0.3081"""
    norm = images.astype(np.float32) / 255.0
    norm = (norm - 0.1307) / 0.3081
    # Expand channel dimension: (N, 1, 28, 28)
    norm = np.expand_dims(norm, axis=1)
    return norm

def benchmark_model(model_path: str, images: np.ndarray, labels: np.ndarray, num_samples: int):
    """Runs inference benchmark on ONNX runtime."""
    print("=" * 70)
    print(f"Loading ONNX Model: {model_path}")
    print(f"Model File Size:   {os.path.getsize(model_path) / 1024:.1f} KB")
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    n = min(num_samples, len(images))
    print(f"Benchmarking on {n:,} test samples...")
    print("=" * 70)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: images[:1]})

    correct = 0
    start_time = time.perf_counter()
    for i in range(n):
        sample = images[i:i+1]
        logits = session.run(None, {input_name: sample})[0]
        pred = np.argmax(logits, axis=1)[0]
        if pred == labels[i]:
            correct += 1

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / n) * 1000.0
    accuracy = (correct / n) * 100.0

    print(f"Accuracy:          {accuracy:.2f}% ({correct}/{n})")
    print(f"Total Time:        {total_time:.3f} s")
    print(f"Average Latency:   {avg_latency_ms:.3f} ms / sample")
    print(f"Throughput:        {n / total_time:.1f} samples / s")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Verify ONNX edge models on MNIST test dataset")
    parser.add_argument("--model", type=str, default="quantized_model_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--samples", type=int, default=10000, help="Number of test samples to benchmark")
    args = parser.parse_args()

    images, labels = load_test_dataset(args.data_dir)
    images_proc = preprocess_mnist(images)
    benchmark_model(args.model, images_proc, labels, args.samples)

if __name__ == "__main__":
    main()
