"""
Raspberry Pi Edge Benchmark Tool for INT8 Quantized CNN
======================================================
Executes high-precision latency profiling, statistical latency distribution
analysis (mean, median, p90, p99, std), hardware telemetry (CPU temp, freq, RSS),
and accuracy evaluation on Raspberry Pi 5 / 4 / CM4.

Usage:
  python benchmark_pi.py --model ../quantized_model_int8.onnx --samples 10000
  python benchmark_pi.py --model ../model_fp32.onnx --samples 10000 --plot
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime is required. Install via: pip install onnxruntime")
    sys.exit(1)


def get_cpu_temperature() -> Optional[float]:
    """Reads Raspberry Pi SoC temperature in Celsius."""
    # Method 1: vcgencmd (Raspberry Pi OS)
    try:
        res = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        if res.returncode == 0 and "temp=" in res.stdout:
            return float(res.stdout.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        pass

    # Method 2: sysfs thermal zone
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_path):
        try:
            with open(thermal_path, "r", encoding="utf-8") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass

    return None


def get_cpu_frequency_mhz() -> Optional[float]:
    """Reads current CPU core clock speed in MHz."""
    freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
    if os.path.exists(freq_path):
        try:
            with open(freq_path, "r", encoding="utf-8") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass
    return None


def get_process_memory_mb() -> float:
    """Returns RSS memory consumption in MB."""
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux ru_maxrss is in KB
        return float(rusage.ru_maxrss) / 1024.0
    except Exception:
        return 0.0


def resolve_model_path(model_arg: str) -> str:
    """Finds model path either locally or relative to repo root."""
    candidates = [
        model_arg,
        os.path.join("models", model_arg),
        os.path.join("..", "models", model_arg),
        os.path.join(os.path.dirname(__file__), "..", "models", model_arg),
        os.path.join("..", model_arg),
        os.path.join(os.path.dirname(__file__), "..", model_arg),
        os.path.join(os.path.dirname(__file__), model_arg),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return model_arg


def resolve_data_path(data_arg: str) -> str:
    """Finds data dir either locally or relative to repo root."""
    candidates = [
        data_arg,
        os.path.join("..", data_arg),
        os.path.join(os.path.dirname(__file__), "..", data_arg),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return os.path.abspath(p)
    return data_arg


def load_mnist_test(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads 10,000 MNIST test samples from npz or ubyte."""
    npz_path = os.path.join(data_dir, "mnist_test_10k.npz")
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        images = data["images"]
        labels = data["labels"]
        return images, labels

    raw_images = os.path.join(data_dir, "MNIST", "raw", "t10k-images-idx3-ubyte")
    raw_labels = os.path.join(data_dir, "MNIST", "raw", "t10k-labels-idx1-ubyte")
    if os.path.isfile(raw_images) and os.path.isfile(raw_labels):
        with open(raw_images, "rb") as f:
            f.read(16)
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        with open(raw_labels, "rb") as f:
            f.read(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return images, labels

    raise FileNotFoundError(f"Could not locate MNIST test dataset in: {data_dir}")


def preprocess(images: np.ndarray) -> np.ndarray:
    """Applies standard MNIST normalization: (x / 255.0 - 0.1307) / 0.3081."""
    arr = images.astype(np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=1)  # (N, 1, 28, 28)
    return arr


def run_benchmark(
    model_path: str,
    images: np.ndarray,
    labels: np.ndarray,
    num_samples: int = 10000,
    warmup: int = 50,
    num_threads: Optional[int] = None,
) -> Dict:
    """Executes full inference benchmark and collects metrics."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if num_threads is not None and num_threads > 0:
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
    else:
        # Default to available physical cores
        cpu_count = os.cpu_count() or 4
        sess_options.intra_op_num_threads = cpu_count
        sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    n = min(num_samples, len(images))
    total_warmup = min(warmup, len(images))

    # Warmup
    for i in range(total_warmup):
        sample = images[i : i + 1]
        session.run(None, {input_name: sample})

    initial_temp = get_cpu_temperature()
    initial_freq = get_cpu_frequency_mhz()

    latencies_ms: List[float] = []
    correct = 0
    cum_acc: List[float] = []

    print(f"\n🚀 Running Benchmark on {n} samples using {session.get_providers()[0]}...")
    start_total = time.perf_counter()

    for i in range(n):
        sample = images[i : i + 1]
        t0 = time.perf_counter()
        logits = session.run(None, {input_name: sample})[0]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

        pred = int(np.argmax(logits, axis=1)[0])
        if pred == labels[i]:
            correct += 1

        if (i + 1) % 100 == 0 or i == n - 1:
            cum_acc.append(correct / (i + 1))

    total_time_sec = time.perf_counter() - start_total
    final_temp = get_cpu_temperature()
    final_freq = get_cpu_frequency_mhz()
    final_rss = get_process_memory_mb()

    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p90_lat = float(np.percentile(lat_arr, 90))
    p99_lat = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    std_lat = float(np.std(lat_arr))
    throughput = n / total_time_sec
    accuracy = (correct / n) * 100.0
    model_size_kb = os.path.getsize(model_path) / 1024.0

    results = {
        "model_path": model_path,
        "model_size_kb": round(model_size_kb, 2),
        "total_samples": n,
        "correct_predictions": correct,
        "total_errors": n - correct,
        "accuracy_pct": round(accuracy, 2),
        "total_time_sec": round(total_time_sec, 3),
        "throughput_fps": round(throughput, 1),
        "latency_mean_ms": round(mean_lat, 4),
        "latency_median_ms": round(median_lat, 4),
        "latency_p90_ms": round(p90_lat, 4),
        "latency_p99_ms": round(p99_lat, 4),
        "latency_min_ms": round(min_lat, 4),
        "latency_max_ms": round(max_lat, 4),
        "latency_std_ms": round(std_lat, 4),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_cores": os.cpu_count(),
            "threads_used": sess_options.intra_op_num_threads,
            "temp_start_c": initial_temp,
            "temp_end_c": final_temp,
            "cpu_freq_mhz": final_freq or initial_freq,
            "memory_rss_mb": round(final_rss, 2),
        },
        "cumulative_accuracy": cum_acc,
    }

    # Print clean formatted banner
    print("=" * 72)
    print(f"  RASPBERRY PI BENCHMARK REPORT: {os.path.basename(model_path)}")
    print("=" * 72)
    print(f"  Model Size:          {model_size_kb:.1f} KB")
    print(f"  Accuracy (Top-1):    {accuracy:.2f}% ({correct}/{n} correct, {n - correct} errors)")
    print(f"  Average Latency:     {mean_lat:.3f} ms / sample")
    print(f"  Median Latency:      {median_lat:.3f} ms / sample")
    print(f"  90th Percentile:     {p90_lat:.3f} ms / sample")
    print(f"  99th Percentile:     {p99_lat:.3f} ms / sample")
    print(f"  Throughput:          {throughput:.1f} samples / s (FPS)")
    print(f"  Total Duration:      {total_time_sec:.3f} s")
    if final_temp is not None:
        print(f"  SoC Temperature:     {final_temp:.1f} °C")
    if final_freq is not None:
        print(f"  CPU Clock:           {final_freq:.0f} MHz")
    print(f"  Peak RSS Memory:     {final_rss:.1f} MB")
    print("=" * 72)

    return results


def plot_accuracy_graph(
    cum_acc: List[float],
    model_name: str,
    save_path: str = "reports/raspberry_pi_accuracy.png",
) -> None:
    """Generates evaluation graph strictly formatted per GEMINI.md standards."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)

        is_fp32 = "fp32" in model_name.lower()
        color = "#1f77b4" if is_fp32 else "#ff7f0e"
        label = "FP32 Baseline" if is_fp32 else "INT8 Quantized"

        ax.plot(range(len(cum_acc)), cum_acc, label=label, color=color, linewidth=1.5)
        ax.set_title("model accuracy", fontsize=11, pad=8)
        ax.set_ylabel("accuracy", fontsize=10)
        ax.set_xlabel("epoch", fontsize=10)
        ax.legend([label], loc="upper left", frameon=True)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(1.0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved evaluation graph to: {save_path}")
    except Exception as e:
        print(f"Graph plot notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge Benchmark for Quantized CNN")
    parser.add_argument("--model", type=str, default="quantized_model_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to MNIST dataset directory")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to evaluate")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--threads", type=int, default=None, help="Number of ONNX Runtime inference threads")
    parser.add_argument("--save-json", type=str, default=None, help="Output JSON results path")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation graph per project standard")
    parser.add_argument("--plot-path", type=str, default="reports/raspberry_pi_accuracy.png", help="Graph save path")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    data_dir = resolve_data_path(args.data_dir)

    print(f"Loading test dataset from: {data_dir}")
    images, labels = load_mnist_test(data_dir)
    images_proc = preprocess(images)

    results = run_benchmark(
        model_path=model_path,
        images=images_proc,
        labels=labels,
        num_samples=args.samples,
        warmup=args.warmup,
        num_threads=args.threads,
    )

    if args.save_json:
        out_dir = os.path.dirname(args.save_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Exclude huge list from clean summary json
        summary = {k: v for k, v in results.items() if k != "cumulative_accuracy"}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved benchmark metrics JSON to: {args.save_json}")

    if args.plot:
        plot_accuracy_graph(
            cum_acc=results["cumulative_accuracy"],
            model_name=os.path.basename(model_path),
            save_path=args.plot_path,
        )


if __name__ == "__main__":
    main()
