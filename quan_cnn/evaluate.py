"""
Accuracy and Latency Benchmarking Module
========================================
Compares FP32 and INT8 models on the full hold-out MNIST test set:
1. Top-1 Classification Accuracy (%)
2. CPU Inference Latency (ms per image & ms per batch)
3. Model File Size and Compression Ratio
"""

from typing import Dict, Any, Tuple
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def benchmark_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    warmup_batches: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates Top-1 accuracy and CPU latency on a given DataLoader.
    
    Args:
        model: Model to evaluate.
        test_loader: Test dataset DataLoader.
        device: Device to execute on.
        warmup_batches: Number of batches to run before measuring latency.
        
    Returns:
        Dict with 'accuracy', 'correct', 'total', 'latency_per_img_ms', 'latency_per_batch_ms'.
    """
    model.eval()
    model.to(device)

    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= warmup_batches:
                break
            images = images.to(device)
            _ = model(images)

    # Benchmark loop
    correct = 0
    total = 0
    batch_latencies = []
    cum_accuracies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            start = time.perf_counter()
            outputs = model(images)
            end = time.perf_counter()

            batch_latencies.append(end - start)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            cum_accuracies.append(correct / total)

    accuracy = (correct / total) * 100.0
    total_time = sum(batch_latencies)
    avg_latency_batch_ms = (total_time / len(batch_latencies)) * 1000.0
    avg_latency_img_ms = (total_time / total) * 1000.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "latency_per_batch_ms": avg_latency_batch_ms,
        "latency_per_img_ms": avg_latency_img_ms,
        "batch_latencies": batch_latencies,
        "cum_accuracies": cum_accuracies,
    }


def plot_quantization_evaluation(
    fp32_results: Dict[str, Any],
    quant_results: Dict[str, Any],
    report_dir: str = "reports",
) -> Tuple[str, str]:
    """
    Plots canonical clean Keras/Matplotlib style evaluation curves for quantization:
      1. model accuracy (FP32 baseline vs INT8 quantized across evaluation batches)
      2. model latency (FP32 baseline vs INT8 quantized per batch)
    Strictly adheres to project formatting standard in GEMINI.md (no grid lines, 4 decimal places).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    os.makedirs(report_dir, exist_ok=True)
    batches = range(len(fp32_results["cum_accuracies"]))

    # 1. Model Accuracy Plot
    acc_path = os.path.join(report_dir, "quantization_accuracy.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    ax.plot(batches, fp32_results["cum_accuracies"], label="FP32", color="#1f77b4", linewidth=1.5)
    ax.plot(batches, quant_results["cum_accuracies"], label="INT8", color="#ff7f0e", linewidth=1.5)
    ax.set_title("model accuracy", fontsize=11, pad=8)
    ax.set_ylabel("accuracy", fontsize=10)
    ax.set_xlabel("batch", fontsize=10)
    ax.legend(["FP32", "INT8"], loc="upper left", frameon=True)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # 2. Model Latency Plot
    lat_path = os.path.join(report_dir, "quantization_latency.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    lat_f = [t * 1000.0 for t in fp32_results["batch_latencies"]]
    lat_q = [t * 1000.0 for t in quant_results["batch_latencies"]]
    ax.plot(batches, lat_f, label="FP32", color="#1f77b4", linewidth=1.5)
    ax.plot(batches, lat_q, label="INT8", color="#ff7f0e", linewidth=1.5)
    ax.set_title("model latency", fontsize=11, pad=8)
    ax.set_ylabel("latency (ms)", fontsize=10)
    ax.set_xlabel("batch", fontsize=10)
    ax.legend(["FP32", "INT8"], loc="upper left", frameon=True)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(lat_path, dpi=300)
    plt.close()

    return acc_path, lat_path


def evaluate_accuracy_and_latency(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    accuracy_budget: float = 0.3,
    report_dir: str = "reports",
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Runs side-by-side benchmark comparing FP32 baseline against INT8 quantized model.
    Checks whether accuracy drop is within the accuracy budget.
    """
    print("\n[Evaluate] Benchmarking FP32 baseline model on full test set...")
    fp32_results = benchmark_model(fp32_model, test_loader, device=device)

    print("[Evaluate] Benchmarking INT8 quantized model on full test set...")
    quant_results = benchmark_model(quant_model, test_loader, device=device)

    acc_drop = fp32_results["accuracy"] - quant_results["accuracy"]
    speedup = fp32_results["latency_per_img_ms"] / max(quant_results["latency_per_img_ms"], 1e-6)
    budget_passed = acc_drop <= accuracy_budget

    # Generate standard evaluation plots (no grid, 4 decimal places)
    acc_plot, lat_plot = plot_quantization_evaluation(fp32_results, quant_results, report_dir=report_dir)

    print("\n" + "=" * 80)
    print("              🏁 FP32 vs. INT8 QUANTIZATION VERIFICATION REPORT")
    print("=" * 80)
    print(f"{'Metric':<28} | {'FP32 Baseline':<16} | {'INT8 Quantized':<16} | {'Delta / Impact'}")
    print("-" * 80)
    fp32_errs = fp32_results["total"] - fp32_results["correct"]
    quant_errs = quant_results["total"] - quant_results["correct"]
    diff_errs = quant_errs - fp32_errs

    print(
        f"{'Top-1 Test Accuracy':<28} | {fp32_results['accuracy']:>14.2f}% | "
        f"{quant_results['accuracy']:>14.2f}% | "
        f"{f'-{acc_drop:.2f}% drop' if acc_drop >= 0 else f'+{-acc_drop:.2f}% gain'} "
        f"({'PASS' if budget_passed else 'FAIL'})"
    )
    print(
        f"{'Total Test Errors (out of 10k)':<28} | {fp32_errs:>14} | "
        f"{quant_errs:>14} | "
        f"{f'+{diff_errs} errors' if diff_errs >= 0 else f'{diff_errs} errors'}"
    )
    print(
        f"{'CPU Latency (per image)':<28} | {fp32_results['latency_per_img_ms']:>13.3f} ms | "
        f"{quant_results['latency_per_img_ms']:>13.3f} ms | "
        f"{speedup:.2f}x speedup"
    )
    print(
        f"{'CPU Latency (per batch)':<28} | {fp32_results['latency_per_batch_ms']:>13.2f} ms | "
        f"{quant_results['latency_per_batch_ms']:>13.2f} ms | "
        f"{speedup:.2f}x speedup"
    )
    print("-" * 80)
    status_str = "🎉 SUCCESS: Accuracy drop is within budget (<= 0.3%)!" if budget_passed else "⚠️ WARNING: Accuracy drop exceeded budget (> 0.3%). Consider QAT."
    print(status_str)
    print(f"\n📈 Saved Quantization Evaluation Graphs (Classic Style):")
    print(f"  • Model Accuracy: {acc_plot}")
    print(f"  • Model Latency:  {lat_plot}")
    print("=" * 80 + "\n")

    return fp32_results, quant_results, budget_passed
