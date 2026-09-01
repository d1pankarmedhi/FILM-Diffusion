#!/usr/bin/env python3
"""
CLI Script for Profiling Hardware & Compute Efficiency (FLOPs, Memory, Latency, Throughput).
"""

import argparse
import os
import torch
from film_diffusion import (
    FiLMDiffusion,
    comprehensive_diffusion_metrics,
    get_device,
    print_comparison_table,
    save_metrics_to_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile FiLM-Diffusion hardware efficiency.")
    parser.add_argument("--base_channels", type=int, nargs="+", default=[32, 64, 128], help="List of base channels to benchmark.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for throughput measurement.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--inference_runs", type=int, default=5, help="Number of timed runs for latency averaging.")
    parser.add_argument("--flops_method", type=str, choices=["fvcore", "thop", "ptflops"], default="fvcore", help="FLOPs profiler library.")
    parser.add_argument("--output_json", type=str, default="diffusion_profiling.json", help="Path to save output JSON metrics.")
    parser.add_argument("--device", type=str, default=None, help="Device to run benchmarks on.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    print("\n" + "=" * 80)
    print(f"⚡ PROFILING FILM-DIFFUSION MODELS ON: {device}")
    print("=" * 80)

    all_metrics = {}

    for bc in args.base_channels:
        model_name = f"FiLMD-{bc}"
        print(f"\n--- Benchmarking {model_name} (base_channels={bc}) ---")

        model = FiLMDiffusion(base_channels=bc, num_timesteps=args.timesteps, device=device).to(device)
        model.eval()

        sample_labels = torch.arange(0, min(10, model.num_classes), device=device)
        sample_shape = (len(sample_labels), model.in_channels, 32, 32)

        metrics = comprehensive_diffusion_metrics(
            model=model,
            sample_fn=model.sample,
            labels=sample_labels,
            shape=sample_shape,
            device=device,
            num_timesteps=args.timesteps,
            guidance_scale=args.guidance_scale,
            num_inference_runs=args.inference_runs,
            flops_method=args.flops_method,
        )
        all_metrics[model_name] = metrics

    # Print summary table
    print_comparison_table(all_metrics)

    # Save to JSON
    save_metrics_to_json(all_metrics, args.output_json)


if __name__ == "__main__":
    main()
