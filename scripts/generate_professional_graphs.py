#!/usr/bin/env python3
"""
Professional Edge AI Benchmark Graph Suite
==========================================
Generates publication-quality, non-clipped benchmark figures for:
1. Raspberry Pi 5 (FP32 Baseline vs. INT8 Quantized)
2. ESP32-S3 Microcontroller (FP32 Baseline vs. INT8 Quantized)
3. Cross-Platform Log-Scale Comparison (Raspberry Pi 5 vs. ESP32-S3)
4. Combined 2-Panel Master Comparative Figure

Strictly adheres to GEMINI.md formatting standards:
- 300 DPI high-resolution output
- Plain white canvas, zero background grid lines
- All 4 bounding box spines visible with consistent linewidth
- Professional typography, clear spacing, zero title/annotation clipping
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)

# Common styling configuration
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 1.2


# ==============================================================================
# Graph 1: Raspberry Pi 5 (FP32 vs. INT8)
# ==============================================================================
def generate_pi5_graph(save_path="reports/raspberry_pi_benchmark.png"):
    if not os.path.isabs(save_path):
        save_path = os.path.join(REPO_ROOT, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    
    categories = ["FP32 Baseline", "INT8 Quantized"]
    latencies = [1.733, 0.282]
    fps = [577.0, 3546.6]
    speedup = latencies[0] / latencies[1]  # 6.15x
    
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(categories, latencies, color=colors, width=0.38, edgecolor="#111111", linewidth=1.2, zorder=3)
    
    ax.set_title("Model Latency Benchmark: Raspberry Pi 5 (ARM Cortex-A76 @ 2.4 GHz)", fontsize=11.5, fontweight="bold", pad=14, color="#111111")
    ax.set_ylabel("Latency (ms / sample)", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylim(0, 2.3)
    
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.06,
            f"{latencies[i]:.3f} ms\n({fps[i]:.0f} FPS)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#111111"
        )
        
    # Badge
    ax.text(
        0.5,
        (latencies[0] + latencies[1]) / 2.0 + 0.25,
        f"{speedup:.2f}x Faster on ARM NEON",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#c92a2a",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#ffffff",
            edgecolor="#c92a2a",
            linewidth=1.5
        ),
        zorder=5
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Generated: {save_path}")


# ==============================================================================
# Graph 2: ESP32-S3 (FP32 vs. INT8)
# ==============================================================================
def generate_esp32_graph(save_path="reports/esp32s3_benchmark.png"):
    if not os.path.isabs(save_path):
        save_path = os.path.join(REPO_ROOT, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    
    categories = ["FP32 Baseline", "INT8 Quantized"]
    latencies = [2820.0, 684.613]
    fps = [0.35, 1.46]
    speedup = latencies[0] / latencies[1]  # 4.12x
    
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(categories, latencies, color=colors, width=0.38, edgecolor="#111111", linewidth=1.2, zorder=3)
    
    ax.set_title("Model Latency Benchmark: ESP32-S3 (Xtensa LX7 @ 240 MHz)", fontsize=11.5, fontweight="bold", pad=14, color="#111111")
    ax.set_ylabel("Latency (ms / sample)", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylim(0, 3600)
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 85,
            f"{latencies[i]:.1f} ms\n({fps[i]:.2f} FPS)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#111111"
        )
        
    # Badge
    ax.text(
        0.5,
        (latencies[0] + latencies[1]) / 2.0,
        f"{speedup:.2f}x Faster on Xtensa LX7",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#c92a2a",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#ffffff",
            edgecolor="#c92a2a",
            linewidth=1.5
        ),
        zorder=5
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Generated: {save_path}")


# ==============================================================================
# Graph 3: Cross-Platform Log-Scale Comparison (Pi 5 vs. ESP32-S3)
# ==============================================================================
def generate_platform_comparison_log(save_path="reports/edge_platform_comparison.png"):
    if not os.path.isabs(save_path):
        save_path = os.path.join(REPO_ROOT, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=300)
    
    platforms = [
        "Raspberry Pi 5\n(ARM Cortex-A76 @ 2.4 GHz)",
        "ESP32-S3 Microcontroller\n(Xtensa LX7 @ 240 MHz)"
    ]
    latencies = [0.282, 684.613]
    fps = [3546.6, 1.46]
    power = ["5.5 W Active (SBC)", "0.35 W Active (MCU)"]
    
    colors = ["#2b8a3e", "#e8590c"]
    bars = ax.bar(platforms, latencies, color=colors, width=0.36, edgecolor="#111111", linewidth=1.2, zorder=3)
    
    ax.set_yscale("log")
    ax.set_ylim(0.04, 7500)
    
    ax.set_title("Edge Inference Latency Comparison: Raspberry Pi 5 vs. ESP32-S3", fontsize=11.5, fontweight="bold", pad=15, color="#111111")
    ax.set_ylabel("Inference Latency (ms / sample, Log Scale)", fontsize=10.5, fontweight="semibold", labelpad=8)
    
    # Custom Log Ticks
    ax.set_yticks([0.1, 1.0, 10.0, 100.0, 1000.0])
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g} ms" if y >= 1 else f"{y:.1f} ms"))
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)
    
    # Text labels over bars
    # For Pi 5 (bar 0)
    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2.0,
        0.282 * 1.35,
        f"0.282 ms\n({fps[0]:.0f} FPS)\n[{power[0]}]",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#111111"
    )
    
    # For ESP32-S3 (bar 1)
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2.0,
        684.613 * 1.25,
        f"684.61 ms\n({fps[1]:.2f} FPS)\n[{power[1]}]",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#111111"
    )
    
    # Center callout badge
    ax.text(
        0.5,
        15.0,
        "Performance vs. Power Trade-Off:\n• Pi 5: 2,428x Faster Throughput (High-Duty SBC)\n• ESP32: 15.7x Lower Power & 0 KB PSRAM (IoT MCU)",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1864ab",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f8f9fa",
            edgecolor="#1864ab",
            linewidth=1.4
        ),
        zorder=5
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Generated: {save_path}")


# ==============================================================================
# Graph 4: Master 2-Panel Figure (Side-by-Side Pi 5 & ESP32-S3)
# ==============================================================================
def generate_combined_master_figure(save_path="reports/edge_benchmarks_combined.png"):
    if not os.path.isabs(save_path):
        save_path = os.path.join(REPO_ROOT, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)
    
    categories = ["FP32 Baseline", "INT8 Quantized"]
    colors = ["#1f77b4", "#ff7f0e"]
    
    # Panel 1: Raspberry Pi 5
    pi_latencies = [1.733, 0.282]
    pi_fps = [577.0, 3546.6]
    bars1 = ax1.bar(categories, pi_latencies, color=colors, width=0.42, edgecolor="#111111", linewidth=1.2, zorder=3)
    ax1.set_title("(a) Raspberry Pi 5 (ARM Cortex-A76 @ 2.4 GHz)", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Latency (ms / sample)", fontsize=10, fontweight="semibold")
    ax1.set_ylim(0, 2.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax1.grid(False)
    
    for i, bar in enumerate(bars1):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.06,
                 f"{pi_latencies[i]:.3f} ms\n({pi_fps[i]:.0f} FPS)",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        
    ax1.text(0.5, 1.20, "6.15x Faster\non ARM NEON", ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="#c92a2a",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor="#c92a2a", linewidth=1.3))
    
    # Panel 2: ESP32-S3
    esp_latencies = [2820.0, 684.613]
    esp_fps = [0.35, 1.46]
    bars2 = ax2.bar(categories, esp_latencies, color=colors, width=0.42, edgecolor="#111111", linewidth=1.2, zorder=3)
    ax2.set_title("(b) ESP32-S3 (Xtensa LX7 @ 240 MHz)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_ylabel("Latency (ms / sample)", fontsize=10, fontweight="semibold")
    ax2.set_ylim(0, 3600)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax2.grid(False)
    
    for i, bar in enumerate(bars2):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 85,
                 f"{esp_latencies[i]:.1f} ms\n({esp_fps[i]:.2f} FPS)",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        
    ax2.text(0.5, 1800, "4.12x Faster\non Xtensa LX7", ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="#c92a2a",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor="#c92a2a", linewidth=1.3))
    
    # Clean spines on both
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(1.2)
        ax.tick_params(axis="both", labelsize=9.5)
        
    fig.suptitle("Edge Quantization Benchmark: Single-Board Computer vs. Microcontroller", fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Generated: {save_path}")


def main():
    generate_pi5_graph()
    generate_esp32_graph()
    generate_platform_comparison_log()
    generate_combined_master_figure()
    print("All professional benchmark graphs successfully generated!")


if __name__ == "__main__":
    main()
