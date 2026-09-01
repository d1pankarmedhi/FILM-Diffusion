#!/usr/bin/env python3
"""
CLI Script for Sampling and Visualizing Generated Images from FiLM-Diffusion.
"""

import argparse
import os
import torch
from film_diffusion import (
    FASHION_CLASSES,
    FiLMDiffusion,
    create_diffusion_gif,
    get_device,
    save_samples_grid,
    set_seed,
    show_samples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples with FiLM-Diffusion.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="Class indices to generate.")
    parser.add_argument("--num_samples_per_class", type=int, default=1, help="Number of images per specified class.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-Free Guidance strength w.")
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddim", help="Sampling algorithm.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference sampling steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity parameter (0=deterministic).")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory to save images.")
    parser.add_argument("--gif", action="store_true", help="Record and generate step-by-step diffusion GIF.")
    parser.add_argument("--gif_save_every", type=int, default=2, help="Capture frequency for GIF frames.")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel width if no checkpoint is provided.")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/mps/cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load or instantiate model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📂 Loading model checkpoint: {args.checkpoint}")
        model, ckpt = FiLMDiffusion.load_from_checkpoint(args.checkpoint, device=device)
    else:
        if args.checkpoint:
            print(f"⚠️  Checkpoint '{args.checkpoint}' not found. Initializing random model.")
        else:
            print("ℹ️  No checkpoint specified. Instantiating fresh model for demonstration.")
        model = FiLMDiffusion(base_channels=args.base_channels, device=device).to(device)

    model.eval()

    # 2. Build target label batch
    labels_list = []
    for c in args.classes:
        labels_list.extend([c] * args.num_samples_per_class)
    labels = torch.tensor(labels_list, device=device, dtype=torch.long)
    batch_size = len(labels)

    class_names = [FASHION_CLASSES.get(int(lbl), f"Class {lbl}") for lbl in labels]
    print(f"🎨 Generating {batch_size} samples: {class_names}")
    print(f"⚡ Sampler: {args.sampler.upper()} ({args.steps} steps) | Guidance: {args.guidance_scale} | Device: {device}")

    # 3. Sampling
    shape = (batch_size, model.in_channels, 32, 32)

    if args.gif:
        print("🎬 Generating progressive trajectory frames...")
        final_samples, intermediates = model.sample_progressive(
            labels=labels,
            shape=shape,
            device=device,
            guidance_scale=args.guidance_scale,
            method=args.sampler,
            num_inference_steps=args.steps,
            eta=args.eta,
            save_every=args.gif_save_every,
        )
        gif_path = os.path.join(args.output_dir, "diffusion_trajectory.gif")
        create_diffusion_gif(intermediates, save_path=gif_path, duration=50)
    else:
        final_samples = model.sample(
            labels=labels,
            shape=shape,
            device=device,
            guidance_scale=args.guidance_scale,
            method=args.sampler,
            num_inference_steps=args.steps,
            eta=args.eta,
            verbose=True,
        )

    # 4. Save results
    grid_path = os.path.join(args.output_dir, "generated_samples_grid.png")
    show_samples(
        final_samples,
        labels=labels,
        title_prefix=f"FiLM-Diffusion {args.sampler.upper()} (CFG={args.guidance_scale})",
        save_path=grid_path,
        display=False,
    )
    print(f"💾 Sample grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
