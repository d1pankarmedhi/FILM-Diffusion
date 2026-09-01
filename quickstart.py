#!/usr/bin/env python3
"""
FiLM-Diffusion Quickstart Demo.
Runs an instant end-to-end verification of model instantiation, loss forward pass, and DDIM sampling.
"""

import time
import torch
from film_diffusion import FiLMDiffusion, FASHION_CLASSES, get_device, show_samples


def main():
    print("=" * 70)
    print("🚀 FiLM-Diffusion Quickstart Demonstration")
    print("=" * 70)

    device = get_device()
    print(f"🖥️  Using execution device: {device}")

    # 1. Model Instantiation
    print("\n[1/3] Instantiating FiLMDiffusion (base_channels=32)...")
    model = FiLMDiffusion(
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        text_emb_dim=128,
        base_channels=32,
        num_timesteps=1000,
        num_classes=10,
        device=device,
    ).to(device)

    total_params = model.num_parameters / 1e6
    print(f"✅ Model created successfully ({total_params:.2f}M parameters)")

    # 2. Forward Pass Verification (Training Step)
    print("\n[2/3] Verifying forward pass with Classifier-Free Guidance dropout...")
    batch_size = 4
    x0 = torch.randn(batch_size, 1, 32, 32, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)

    loss, pred_noise, true_noise, xt = model(x0, labels, cfg_dropout=0.1)
    print(f"✅ Forward pass MSE Loss: {loss.item():.4f}")
    print(f"   Input shape: {x0.shape} -> Predicted noise shape: {pred_noise.shape}")

    # 3. Fast Sampling Verification (DDIM)
    print("\n[3/3] Testing fast DDIM sampling (20 steps)...")
    sample_labels = torch.tensor([0, 3, 7, 8], device=device)  # T-shirt, Dress, Sneaker, Bag
    prompt_names = [FASHION_CLASSES[int(lbl)] for lbl in sample_labels]
    print(f"   Target classes: {prompt_names}")

    start_t = time.time()
    samples = model.sample(
        labels=sample_labels,
        shape=(len(sample_labels), 1, 32, 32),
        device=device,
        guidance_scale=7.5,
        method="ddim",
        num_inference_steps=20,
        eta=0.0,
        verbose=True,
    )
    elapsed = time.time() - start_t
    print(f"✅ Generated {len(samples)} samples in {elapsed:.2f}s! Output shape: {samples.shape}")

    save_path = "quickstart_demo.png"
    show_samples(
        samples,
        labels=sample_labels,
        title_prefix="FiLM-Diffusion Quickstart (DDIM 20 Steps)",
        save_path=save_path,
        display=False,
    )
    print(f"💾 Sample grid saved to: {save_path}")

    print("\n" + "=" * 70)
    print("🎉 All checks passed successfully! FiLM-Diffusion is ready to use.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
