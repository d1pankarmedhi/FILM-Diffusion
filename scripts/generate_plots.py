#!/usr/bin/env python3
"""
Script to Recreate All Research Paper Figures and Ablation Plots.
Outputs high-resolution vector PDFs and PNG figures.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib_style():
    """Configure professional IEEE/CVPR publication styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.2,
        "figure.dpi": 300,
    })


def plot_guidance_scale_ablation(output_dir: str):
    """Plot Effect of Classifier-Free Guidance (CFG) Scale on Metrics."""
    scales = np.array([1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0])
    fid = np.array([35.95, 31.73, 33.99, 41.46, 52.74, 62.36, 80.08])
    is_score = np.array([3.95, 3.98, 3.96, 3.89, 3.83, 3.70, 3.77])
    kid = np.array([11.73, 4.16, 3.18, 4.70, 10.07, 17.47, 30.40])
    precision = np.array([0.425, 0.568, 0.610, 0.582, 0.509, 0.361, 0.263])
    recall = np.array([0.739, 0.817, 0.772, 0.636, 0.502, 0.360, 0.204])

    color_fid = "#e74c3c"
    color_is = "#3498db"
    color_kid = "#2c3e50"
    color_precision = "#16a085"
    color_recall = "#8e44ad"

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    ax1.plot(scales, fid, marker="o", linestyle="-", color=color_fid, label="FID (↓)", markersize=7)
    ax1.plot(scales, kid, marker="s", linestyle="--", color=color_kid, label="KID × 10⁻³ (↓)", markersize=7)
    ax1.plot(scales, is_score, marker="D", linestyle="-.", color=color_is, label="IS (↑)", markersize=7)

    ax1.set_xlabel("Classifier-Free Guidance Scale ($w$)")
    ax1.set_ylabel("FID / KID / IS")
    ax1.set_xticks(scales)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(scales, precision, marker="^", linestyle="-", color=color_precision, label="Precision (↑)", markersize=7)
    ax2.plot(scales, recall, marker="v", linestyle="-", color=color_recall, label="Recall (↑)", markersize=7)
    ax2.set_ylabel("Precision / Recall")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    plt.title("Effect of Guidance Scale on Generation Quality")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "guidance_vs_metrics.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "guidance_vs_metrics.png"), bbox_inches="tight")
    plt.close()
    print(f"✓ Generated: guidance_vs_metrics.png in {output_dir}")


def plot_memory_comparison(output_dir: str):
    """Plot Memory Consumption and Efficiency Comparisons across Base Channel Multipliers."""
    models = ["32", "64", "96", "128"]
    x = np.arange(len(models))

    cad_memory = np.array([1377.98, 3641.89, 3840.48, 5150.03])
    fd_memory = np.array([695.81, 1621.31, 2640.50, 2852.77])
    cad_params = np.array([2.08, 7.41, 16.17, 28.37])
    fd_params = np.array([1.85, 6.48, 14.08, 24.66])

    color_cad = "#e74c3c"
    color_fd = "#3498db"

    # 1. Bar Chart
    plt.figure(figsize=(8.5, 5.5))
    width = 0.35
    b1 = plt.bar(x - width / 2, cad_memory, width, label="Cross-Attention", color=color_cad, alpha=0.9, edgecolor="black")
    b2 = plt.bar(x + width / 2, fd_memory, width, label="FiLM-Diffusion (Ours)", color=color_fd, alpha=0.9, edgecolor="black")

    plt.xlabel("Base Channels")
    plt.ylabel("Peak VRAM Memory (MB)")
    plt.xticks(x, models)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.title("Peak Memory Consumption Comparison")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 50, f"{h:.0f}", ha="center", fontsize=10)

    # Savings annotation
    for i in range(len(models)):
        savings = (cad_memory[i] - fd_memory[i]) / cad_memory[i] * 100
        mid = (cad_memory[i] + fd_memory[i]) / 2
        plt.annotate(
            f"-{savings:.1f}%",
            xy=(i, mid),
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", alpha=0.4),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "memory_comparison.png"), bbox_inches="tight")
    plt.close()

    # 2. Dual Axis: Memory and Parameter Count
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(x, cad_memory, marker="o", color=color_cad, label="Memory (Cross-Attn)", markersize=8)
    ax1.plot(x, fd_memory, marker="s", color=color_fd, linestyle="--", label="Memory (FiLM)", markersize=8)
    ax1.set_xlabel("Base Channels")
    ax1.set_ylabel("Memory (MB)", color=color_cad)
    ax1.tick_params(axis="y", labelcolor=color_cad)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(x, cad_params, marker="^", color="#2ecc71", label="Params (Cross-Attn)", markersize=8)
    ax2.plot(x, fd_params, marker="v", color="#16a085", linestyle="--", label="Params (FiLM)", markersize=8)
    ax2.set_ylabel("Parameters (Millions)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    plt.title("Memory Consumption and Model Capacity Scaling")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_params_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "memory_params_comparison.png"), bbox_inches="tight")
    plt.close()

    print(f"✓ Generated: memory_comparison.png and memory_params_comparison.png in {output_dir}")


def plot_quality_metrics(output_dir: str):
    """Plot IS + KID and Precision + Recall across model scales."""
    models = ["64", "96", "128"]
    x = np.arange(len(models))

    cad_is = np.array([4.16, 3.85, 3.99])
    fd_is = np.array([4.16, 3.85, 3.99])
    cad_kid = np.array([9.07, 8.76, 8.94])
    fd_kid = np.array([9.03, 8.76, 8.94])

    cad_precision = np.array([0.781, 0.805, 0.823])
    fd_precision = np.array([0.778, 0.802, 0.819])
    cad_recall = np.array([0.652, 0.628, 0.602])
    fd_recall = np.array([0.658, 0.635, 0.611])

    color_cad = "#e74c3c"
    color_fd = "#3498db"

    # 1. IS + KID
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(x, cad_is, marker="o", color=color_cad, label="Cross-Attn: IS (↑)", markersize=8)
    ax1.plot(x, fd_is, marker="s", color=color_fd, linestyle="--", label="FiLM: IS (↑)", markersize=8)
    ax1.set_ylabel("Inception Score (IS ↑)")
    ax1.set_xlabel("Base Channels")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(x, cad_kid, marker="^", color="#f39c12", label="Cross-Attn: KID (↓)", markersize=8)
    ax2.plot(x, fd_kid, marker="v", color="#27ae60", linestyle="--", label="FiLM: KID (↓)", markersize=8)
    ax2.set_ylabel("KID × 10⁻³ (↓)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=False, ncol=2)
    plt.title("Image Quality Assessment (IS & KID)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "is_kid_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "is_kid_comparison.png"), bbox_inches="tight")
    plt.close()

    # 2. Precision + Recall
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(x, cad_precision, marker="o", color=color_cad, label="Cross-Attn: Precision (↑)", markersize=8)
    ax1.plot(x, fd_precision, marker="s", color=color_fd, linestyle="--", label="FiLM: Precision (↑)", markersize=8)
    ax1.set_ylabel("Precision (Fidelity ↑)")
    ax1.set_xlabel("Base Channels")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(x, cad_recall, marker="^", color="#8e44ad", label="Cross-Attn: Recall (↑)", markersize=8)
    ax2.plot(x, fd_recall, marker="v", color="#16a085", linestyle="--", label="FiLM: Recall (↑)", markersize=8)
    ax2.set_ylabel("Recall (Diversity ↑)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=False, ncol=2)
    plt.title("Quality vs. Diversity (Precision & Recall)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "precision_recall_comparison.png"), bbox_inches="tight")
    plt.close()

    print(f"✓ Generated: is_kid_comparison.png and precision_recall_comparison.png in {output_dir}")


def plot_throughput_comparison(output_dir: str):
    """Plot Inference Throughput (samples/sec) across models."""
    models = ["32", "64", "96", "128"]
    x = np.arange(len(models))

    # Throughput (samples per second)
    cad_throughput = np.array([24.2, 13.2, 7.8, 4.2])
    fd_throughput = np.array([68.5, 42.3, 24.8, 14.1])

    color_cad = "#e74c3c"
    color_fd = "#3498db"

    plt.figure(figsize=(8.5, 5.5))
    width = 0.35
    b1 = plt.bar(x - width / 2, cad_throughput, width, label="Cross-Attention", color=color_cad, alpha=0.9, edgecolor="black")
    b2 = plt.bar(x + width / 2, fd_throughput, width, label="FiLM-Diffusion (Ours)", color=color_fd, alpha=0.9, edgecolor="black")

    plt.xlabel("Base Channels")
    plt.ylabel("Inference Throughput (samples / sec ↑)")
    plt.xticks(x, models)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.title("Inference Throughput (Higher is Better)")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", fontsize=10)

    # Speedup annotations
    for i in range(len(models)):
        speedup = fd_throughput[i] / cad_throughput[i]
        plt.annotate(
            f"{speedup:.1f}×",
            xy=(i + width / 2, fd_throughput[i] / 2),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"), bbox_inches="tight")
    plt.close()
    print(f"✓ Generated: throughput_comparison.png in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Directory to save figures.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_matplotlib_style()

    print("📊 Generating publication figures...")
    plot_guidance_scale_ablation(args.output_dir)
    plot_memory_comparison(args.output_dir)
    plot_quality_metrics(args.output_dir)
    plot_throughput_comparison(args.output_dir)
    print("🎉 All figures generated successfully!")


if __name__ == "__main__":
    main()
