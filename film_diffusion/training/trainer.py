"""
Training Engine for Text/Class-Conditional FiLM Diffusion Models.
Supports Automatic Mixed Precision (AMP), Gradient Accumulation, Cosine Annealing,
and robust Checkpoint Resumption.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from film_diffusion.models.diffusion import FiLMDiffusion
from film_diffusion.utils.visualization import show_samples


class Trainer:
    """
    Production-grade Trainer for FiLMDiffusion models.

    Args:
        model: FiLMDiffusion instance.
        dataloader: Training DataLoader.
        timesteps: Number of diffusion timesteps (default: 1000).
        num_epochs: Total number of training epochs (default: 50).
        lr: Peak learning rate (default: 1e-4).
        weight_decay: AdamW weight decay (default: 1e-4).
        cfg_dropout: Classifier-Free Guidance training dropout probability (default: 0.1).
        guidance_scale: Guidance scale used during validation sampling (default: 7.5).
        checkpoint_dir: Directory where checkpoints and sample grids are stored.
        device: Device string or torch.device ('cuda' or 'cpu').
        use_amp: Whether to enable Automatic Mixed Precision (default: True).
        gradient_accumulation_steps: Gradient accumulation steps (default: 1).
        sample_interval: Epoch frequency to generate validation samples (default: 5).
        max_grad_norm: Maximum norm for gradient clipping (default: 1.0).
    """

    def __init__(
        self,
        model: FiLMDiffusion,
        dataloader: DataLoader,
        timesteps: int = 1000,
        num_epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        cfg_dropout: float = 0.1,
        guidance_scale: float = 7.5,
        checkpoint_dir: str = "./checkpoints",
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        sample_interval: int = 5,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.timesteps = timesteps
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.cfg_dropout = cfg_dropout
        self.guidance_scale = guidance_scale
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.sample_interval = sample_interval
        self.max_grad_norm = max_grad_norm

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.eval_dir = os.path.join(self.checkpoint_dir, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)

        self.model.to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.lr * 0.01,
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Tracking variables
        self.start_epoch = 0
        self.training_losses: List[float] = []
        self.metrics_history: List[Dict] = []
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []

    def resume_from_checkpoint(self, checkpoint_path: Union[str, None] = "latest") -> bool:
        """
        Resume training state from a checkpoint path or find the latest checkpoint in checkpoint_dir.

        Returns:
            True if successfully loaded, False otherwise.
        """
        if not checkpoint_path:
            return False

        if checkpoint_path == "latest":
            if not os.path.exists(self.checkpoint_dir):
                return False
            checkpoints = [
                f for f in os.listdir(self.checkpoint_dir)
                if f.startswith(f"model_weight_timesteps_{self.timesteps}_epoch_") and f.endswith(".pth")
            ]
            if not checkpoints:
                # Try generic matching
                checkpoints = [
                    f for f in os.listdir(self.checkpoint_dir)
                    if f.startswith("model_weight_") and f.endswith(".pth") and "epoch_" in f
                ]
            if not checkpoints:
                print("ℹ️  No existing checkpoints found. Starting from scratch.")
                return False

            epochs = [int(f.split("epoch_")[1].split(".pth")[0]) for f in checkpoints]
            latest_epoch = max(epochs)
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_weight_timesteps_{self.timesteps}_epoch_{latest_epoch}.pth"
            )
            if not os.path.exists(checkpoint_path):
                # Fallback to the first matched file
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[0])

        if os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])

            self.start_epoch = checkpoint.get("epoch", 0)
            self.training_losses = checkpoint.get("training_losses", [])
            self.metrics_history = checkpoint.get("metrics_history", [])
            self.cfg_dropout = checkpoint.get("cfg_dropout", self.cfg_dropout)
            self.guidance_scale = checkpoint.get("guidance_scale", self.guidance_scale)

            print(f"✅ Resumed from epoch {self.start_epoch} (losses logged: {len(self.training_losses)})")
            return True
        else:
            print(f"⚠️  Checkpoint path not found: {checkpoint_path}")
            return False

    def train(self, resume: Optional[Union[str, bool]] = None) -> Tuple[FiLMDiffusion, List[float], Dict]:
        """
        Execute full training loop.

        Args:
            resume: Path to checkpoint, 'latest', or True to resume from latest checkpoint.

        Returns:
            Tuple of (trained_model, training_losses_list, timing_info_dict)
        """
        if resume:
            ckpt_arg = "latest" if resume is True else resume
            self.resume_from_checkpoint(ckpt_arg)

        print("\n" + "=" * 70)
        print(f"🚀 Starting FiLM-Diffusion Training (Epochs {self.start_epoch + 1} to {self.num_epochs})")
        print(f"⚙️  Timesteps: {self.timesteps} | Base Channels: {self.model.base_channels} | CFG Dropout: {self.cfg_dropout}")
        print(f"⚡ Device: {self.device} | AMP: {self.use_amp} | Grad Accum: {self.gradient_accumulation_steps}")
        print(f"💾 Checkpoints directory: {self.checkpoint_dir}")
        print("=" * 70 + "\n")

        total_training_start = time.time()

        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            epoch_losses = []

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch_idx, (images, labels) in enumerate(pbar):
                batch_start = time.time()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        loss, _, _, _ = self.model(images, labels, cfg_dropout=self.cfg_dropout)
                        loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    loss, _, _, _ = self.model(images, labels, cfg_dropout=self.cfg_dropout)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                # Optimization step with gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.dataloader):
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                scaled_loss = loss.item() * self.gradient_accumulation_steps
                epoch_losses.append(scaled_loss)
                self.batch_times.append(time.time() - batch_start)

                pbar.set_postfix({
                    "loss": f"{scaled_loss:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                })

            pbar.close()

            # Epoch summary
            avg_epoch_loss = float(np.mean(epoch_losses))
            self.training_losses.append(avg_epoch_loss)
            self.scheduler.step()

            epoch_elapsed = time.time() - epoch_start_time
            self.epoch_times.append(epoch_elapsed)
            print(f"📈 Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f} (took {epoch_elapsed:.1f}s)")

            # Periodic validation sampling & checkpoint saving
            if (epoch + 1) % self.sample_interval == 0 or (epoch + 1) == self.num_epochs:
                self._save_checkpoint_and_samples(epoch + 1, avg_epoch_loss)

        total_training_time = time.time() - total_training_start
        print(f"\n🎉 Training completed in {total_training_time / 60:.2f} minutes!\n")

        timing_info = {
            "total_training_time_s": total_training_time,
            "epoch_times_s": self.epoch_times,
            "avg_epoch_time_s": float(np.mean(self.epoch_times)) if self.epoch_times else 0.0,
            "avg_batch_time_s": float(np.mean(self.batch_times)) if self.batch_times else 0.0,
            "throughput_samples_per_sec": (len(self.batch_times) * self.dataloader.batch_size / total_training_time) if total_training_time > 0 else 0.0,
        }

        return self.model, self.training_losses, timing_info

    def _save_checkpoint_and_samples(self, epoch: int, current_loss: float) -> None:
        """Helper to generate sample grids and save checkpoint file."""
        self.model.eval()

        # 1. Generate DDPM samples
        with torch.no_grad():
            sample_labels = torch.arange(0, min(10, self.model.num_classes), device=self.device)
            sample_shape = (len(sample_labels), self.model.in_channels, 32, 32)

            ddpm_samples = self.model.sample(
                labels=sample_labels,
                shape=sample_shape,
                device=self.device,
                guidance_scale=self.guidance_scale,
                method="ddpm",
                num_inference_steps=self.timesteps,
            )
            sample_path = os.path.join(self.eval_dir, f"samples_epoch_{epoch}.png")
            show_samples(
                ddpm_samples,
                epoch=epoch,
                labels=sample_labels,
                title_prefix=f"DDPM (CFG={self.guidance_scale})",
                save_path=sample_path,
                display=False,
            )

            # 2. Generate Fast DDIM samples (50 steps)
            ddim_samples = self.model.sample(
                labels=sample_labels,
                shape=sample_shape,
                device=self.device,
                guidance_scale=self.guidance_scale,
                method="ddim",
                num_inference_steps=min(50, self.timesteps),
            )
            ddim_sample_path = os.path.join(self.eval_dir, f"ddim_samples_epoch_{epoch}.png")
            show_samples(
                ddim_samples,
                epoch=epoch,
                labels=sample_labels,
                title_prefix=f"DDIM (CFG={self.guidance_scale}, 50 steps)",
                save_path=ddim_sample_path,
                display=False,
            )

        # 3. Save checkpoint
        ckpt_path = os.path.join(
            self.checkpoint_dir, f"model_weight_timesteps_{self.timesteps}_epoch_{epoch}.pth"
        )
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "loss": current_loss,
            "training_losses": self.training_losses,
            "cfg_dropout": self.cfg_dropout,
            "guidance_scale": self.guidance_scale,
            "metrics_history": self.metrics_history,
            "config": {
                "in_channels": self.model.in_channels,
                "out_channels": self.model.out_channels,
                "time_emb_dim": self.model.time_emb_dim,
                "text_emb_dim": self.model.text_emb_dim,
                "base_channels": self.model.base_channels,
                "num_timesteps": self.timesteps,
                "num_classes": self.model.num_classes,
            },
        }, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")


def train_text_conditional_diffusion(
    model: FiLMDiffusion,
    dataloader: DataLoader,
    timesteps: int = 1000,
    num_epochs: int = 50,
    lr: float = 1e-4,
    cfg_dropout: float = 0.1,
    guidance_scale: float = 7.5,
    sample_interval: int = 5,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    **kwargs,
) -> Tuple[FiLMDiffusion, List[float], Dict]:
    """
    Functional wrapper for training FiLMDiffusion models (compatible with original notebook API).
    """
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        timesteps=timesteps,
        num_epochs=num_epochs,
        lr=lr,
        cfg_dropout=cfg_dropout,
        guidance_scale=guidance_scale,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_amp=use_amp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sample_interval=sample_interval,
    )
    return trainer.train(resume=resume_from_checkpoint)
