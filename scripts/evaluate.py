#!/usr/bin/env python3
"""
CLI Script for Quantitative Evaluation of FiLM-Diffusion (FID, IS, KID, Precision, Recall).
"""

import argparse
import os
import torch
from film_diffusion import (
    FiLMDiffusion,
    evaluate_metrics_with_precision_recall,
    get_dataset,
    get_device,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FiLM-Diffusion generation quality metrics.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth).")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", help="Benchmark dataset (fashion_mnist, mnist).")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where dataset is stored.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of real/fake samples for metric computation.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Sampling guidance scale.")
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddim", help="Sampling algorithm.")
    parser.add_argument("--steps", type=int, default=50, help="Sampling inference steps.")
    parser.add_argument("--nearest_k", type=int, default=3, help="k-nearest neighbor parameter for Precision/Recall.")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Directory to store evaluation results.")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    # 1. Load model
    print(f"📂 Loading model checkpoint: {args.checkpoint}")
    model, _ = FiLMDiffusion.load_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # 2. Load test set
    print(f"📦 Loading evaluation dataset: {args.dataset}")
    _, test_dataset, _, test_loader = get_dataset(
        dataset_name=args.dataset,
        batch_size=128,
        root=args.data_dir,
    )

    # 3. Run evaluation
    results = evaluate_metrics_with_precision_recall(
        model=model,
        real_dataloader=test_loader,
        device=str(device),
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        nearest_k=args.nearest_k,
        save_dir=args.output_dir,
        method=args.sampler,
        num_inference_steps=args.steps,
    )


if __name__ == "__main__":
    main()
