#!/usr/bin/env python3
"""
CLI Script for Training FiLM-Diffusion Models.
"""

import argparse
import sys
from film_diffusion import FiLMDiffusion, Trainer, get_dataset, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train FiLM-Diffusion on image benchmarks.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", help="Dataset (fashion_mnist, mnist, cifar10).")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where dataset is stored.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--base_channels", type=int, default=64, help="UNet base channel width.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total diffusion timesteps.")
    parser.add_argument("--cfg_dropout", type=float, default=0.1, help="Classifier-Free Guidance training dropout.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Sampling guidance scale.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to store model weights.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'latest' to resume.")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--sample_interval", type=int, default=5, help="Epoch interval to generate validation samples.")
    parser.add_argument("--device", type=str, default=None, help="Device to run training on (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Base configuration with optional YAML override
    cfg = {}
    if args.config:
        print(f"📄 Loading config from: {args.config}")
        cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    base_channels = model_cfg.get("base_channels", args.base_channels)
    timesteps = model_cfg.get("num_timesteps", args.timesteps)
    num_classes = model_cfg.get("num_classes", 10)
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", 1)

    dataset_name = train_cfg.get("dataset", args.dataset)
    epochs = train_cfg.get("num_epochs", args.epochs)
    batch_size = train_cfg.get("batch_size", args.batch_size)
    lr = float(train_cfg.get("lr", args.lr))
    cfg_dropout = float(train_cfg.get("cfg_dropout", args.cfg_dropout))
    guidance_scale = float(train_cfg.get("guidance_scale", args.guidance_scale))
    checkpoint_dir = train_cfg.get("checkpoint_dir", args.checkpoint_dir)
    use_amp = not args.no_amp if args.no_amp else train_cfg.get("use_amp", True)
    grad_accum = train_cfg.get("gradient_accumulation_steps", args.grad_accum)
    sample_interval = train_cfg.get("sample_interval", args.sample_interval)

    # 1. Load Dataset
    print(f"📦 Loading dataset: {dataset_name} (batch_size={batch_size})")
    train_ds, test_ds, train_loader, test_loader = get_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        root=args.data_dir,
    )

    # 2. Instantiate Model
    model = FiLMDiffusion(
        in_channels=in_channels,
        out_channels=out_channels,
        time_emb_dim=128,
        text_emb_dim=128,
        base_channels=base_channels,
        num_timesteps=timesteps,
        num_classes=num_classes,
    )
    print(f"🧠 Model instantiated: {model.num_parameters / 1e6:.2f}M parameters")

    # 3. Train
    device = args.device or ("cuda" if sys.platform != "darwin" and args.device != "cpu" else "cpu")
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        timesteps=timesteps,
        num_epochs=epochs,
        lr=lr,
        cfg_dropout=cfg_dropout,
        guidance_scale=guidance_scale,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_amp=use_amp,
        gradient_accumulation_steps=grad_accum,
        sample_interval=sample_interval,
    )

    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
