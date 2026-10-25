#!/usr/bin/env python3
"""
ESP32-S3 Latency Benchmark Plotter
==================================
Generates high-resolution publication-quality benchmark graph for ESP32-S3:
Model Latency Benchmark: ESP32-S3 (Xtensa LX7 @ 240 MHz): FP32 Baseline vs. INT8 Quantized.

Strictly adheres to GEMINI.md formatting standards:
- 300 DPI high-resolution output
- Plain white canvas, zero background grid lines
- All 4 bounding box spines visible
- Non-clipped title and bold annotations
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_esp32_latency(save_path="reports/esp32s3_benchmark.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = ["FP32 Baseline", "INT8 Quantized"]
    latencies = [2820.0, 684.613]  # in ms
    fps = [0.35, 1.46]
    speedup = latencies[0] / latencies[1]  # 4.12x

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)

    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, latencies, color=colors, width=0.38, edgecolor="#111111", linewidth=1.2, zorder=3)

    ax.set_title("Model Latency Benchmark: ESP32-S3 (Xtensa LX7 @ 240 MHz)", fontsize=11.5, fontweight="bold", pad=14, color="#111111")
    ax.set_ylabel("Latency (ms / sample)", fontsize=10.5, fontweight="semibold", labelpad=8)
    ax.set_ylim(0, 3600)

    # Format Y axis without excessive decimals
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Remove grid lines and ensure all 4 spines visible
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)

    # Bar value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 85,
            f"{latencies[i]:.1f} ms\n({fps[i]:.2f} FPS)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#111111",
        )

    # Center callout badge: Speedup on Xtensa LX7
    badge_x = 0.5
    badge_y = (latencies[0] + latencies[1]) / 2.0
    ax.text(
        badge_x,
        badge_y,
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
            linewidth=1.5,
        ),
        zorder=5,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved professional ESP32-S3 benchmark plot to: {save_path}")

if __name__ == "__main__":
    plot_esp32_latency()
