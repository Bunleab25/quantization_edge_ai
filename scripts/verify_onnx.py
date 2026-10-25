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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resolve_model_path(model_arg: str) -> str:
    candidates = [
        model_arg,
        os.path.join(REPO_ROOT, "models", os.path.basename(model_arg)),
        os.path.join(REPO_ROOT, model_arg),
        os.path.join("models", model_arg),
        os.path.join("..", "models", model_arg),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return model_arg

def load_test_dataset(data_dir: str = "data"):
    """Loads 10,000 MNIST test images and labels from npz or raw ubyte."""
    candidates = [
        data_dir,
        os.path.join(REPO_ROOT, data_dir),
        os.path.join("..", data_dir),
    ]
    target_dir = data_dir
    for c in candidates:
        if os.path.isdir(c):
            target_dir = c
            break

    npz_path = os.path.join(target_dir, "mnist_test_10k.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data["images"], data["labels"]

    raw_img = os.path.join(target_dir, "MNIST", "raw", "t10k-images-idx3-ubyte")
    raw_lbl = os.path.join(target_dir, "MNIST", "raw", "t10k-labels-idx1-ubyte")
    if os.path.exists(raw_img) and os.path.exists(raw_lbl):
        with open(raw_img, "rb") as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with open(raw_lbl, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return images, labels

    raise FileNotFoundError(f"Test dataset not found in {data_dir} or {target_dir}!")

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
    cum_acc = []
    start_time = time.perf_counter()
    for i in range(n):
        sample = images[i:i+1]
        logits = session.run(None, {input_name: sample})[0]
        pred = np.argmax(logits, axis=1)[0]
        if pred == labels[i]:
            correct += 1
        if (i + 1) % 100 == 0 or i == n - 1:
            cum_acc.append(correct / (i + 1))

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / n) * 1000.0
    accuracy = (correct / n) * 100.0

    print(f"Accuracy:          {accuracy:.2f}% ({correct}/{n})")
    print(f"Total Time:        {total_time:.3f} s")
    print(f"Average Latency:   {avg_latency_ms:.3f} ms / sample")
    print(f"Throughput:        {n / total_time:.1f} samples / s")
    print("=" * 70)
    return cum_acc


def plot_onnx_accuracy_graph(
    cum_acc: list,
    model_label: str = "INT8",
    save_path: str = "reports/onnx_quantization_accuracy.png",
):
    """Saves clean classic evaluation graph strictly following GEMINI.md standards."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        if not os.path.isabs(save_path):
            save_path = os.path.join(REPO_ROOT, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
        ax.plot(range(len(cum_acc)), cum_acc, label=model_label, color="#1f77b4" if "FP32" in model_label else "#ff7f0e", linewidth=1.5)
        ax.set_title("model accuracy", fontsize=11, pad=8)
        ax.set_ylabel("accuracy", fontsize=10)
        ax.set_xlabel("epoch", fontsize=10)
        ax.legend([model_label], loc="upper left", frameon=True)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved evaluation graph to: {save_path}")
    except Exception as e:
        print(f"Plot note: {e}")


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX edge models on MNIST test dataset")
    parser.add_argument("--model", type=str, default="models/quantized_model_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--samples", type=int, default=10000, help="Number of test samples to benchmark")
    parser.add_argument("--plot", action="store_true", help="Generate classic style evaluation graph")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    images, labels = load_test_dataset(args.data_dir)
    images_proc = preprocess_mnist(images)
    cum_acc = benchmark_model(model_path, images_proc, labels, args.samples)

    if args.plot:
        label = "FP32" if "fp32" in model_path.lower() else "INT8"
        plot_onnx_accuracy_graph(cum_acc, model_label=label)


if __name__ == "__main__":
    main()
