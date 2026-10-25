#!/usr/bin/env python3
"""
Cross-Platform Edge Comparison Plotter: Raspberry Pi 5 vs. ESP32-S3
===================================================================
Plots log-scale comparison chart contrasting INT8 quantized latency
and throughput across Raspberry Pi 5 (Edge SBC) and ESP32-S3 (Edge MCU).

Strictly adheres to GEMINI.md formatting standards:
- 300 DPI high-resolution output
- Plain white canvas, zero background grid lines
- All 4 bounding box spines visible
- Non-clipped title and bold annotations
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def plot_cross_platform_comparison(save_path="reports/edge_platform_comparison.png"):
    if not os.path.isabs(save_path):
        save_path = os.path.join(REPO_ROOT, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=300)

    platforms = [
        "Raspberry Pi 5\n(ARM Cortex-A76 @ 2.4 GHz)",
        "ESP32-S3 Microcontroller\n(Xtensa LX7 @ 240 MHz)"
    ]
    latencies = [0.282, 684.613]  # in ms
    fps = [3546.6, 1.46]
    power = ["5.5 W Active (SBC)", "0.35 W Active (MCU)"]

    colors = ["#2b8a3e", "#e8590c"]  # Green for high-perf SBC, Deep orange for ultra-low-power MCU
    bars = ax.bar(platforms, latencies, color=colors, width=0.36, edgecolor="#111111", linewidth=1.2, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(0.04, 7500)

    ax.set_title("Edge Inference Latency Comparison: Raspberry Pi 5 vs. ESP32-S3", fontsize=11.5, fontweight="bold", pad=15, color="#111111")
    ax.set_ylabel("Inference Latency (ms / sample, Log Scale)", fontsize=10.5, fontweight="semibold", labelpad=8)

    # Custom Log Ticks
    ax.set_yticks([0.1, 1.0, 10.0, 100.0, 1000.0])
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g} ms" if y >= 1 else f"{y:.1f} ms"))
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Remove grid lines and ensure all 4 spines visible
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)

    # Annotations over bars
    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2.0,
        0.282 * 1.35,
        f"0.282 ms\n({fps[0]:.0f} FPS)\n[{power[0]}]",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#111111",
    )

    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2.0,
        684.613 * 1.25,
        f"684.61 ms\n({fps[1]:.2f} FPS)\n[{power[1]}]",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#111111",
    )

    # Center callout badge: Tradeoff description
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
            linewidth=1.4,
        ),
        zorder=5,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved professional cross-platform comparison plot to: {save_path}")

if __name__ == "__main__":
    plot_cross_platform_comparison()
