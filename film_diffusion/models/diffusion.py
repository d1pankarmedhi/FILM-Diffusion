"""
FiLMDiffusion: High-Level Text/Class-Conditional Diffusion Model Wrapper with CFG.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from film_diffusion.models.scheduler import NoiseScheduler
from film_diffusion.models.unet import UNetWithFiLM


class FiLMDiffusion(nn.Module):
    """
    Complete Text/Class-Conditional Diffusion Model with Feature-wise Linear Modulation (FiLM)
    and Classifier-Free Guidance (CFG).

    Args:
        in_channels (int): Input image channels (default: 1).
        out_channels (int): Output image channels (default: 1).
        time_emb_dim (int): Timestep embedding dimension (default: 128).
        text_emb_dim (int): Text/label conditioning embedding dimension (default: 128).
        base_channels (int): UNet base channels (default: 64).
        num_timesteps (int): Number of diffusion training timesteps (default: 1000).
        num_classes (int): Number of conditioning classes (default: 10).
        device (str or torch.device): Device on which to initialize.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_emb_dim: int = 128,
        text_emb_dim: int = 128,
        base_channels: int = 64,
        num_timesteps: int = 1000,
        num_classes: int = 10,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim
        self.base_channels = base_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.null_token = num_classes  # Reserved index for unconditional CFG token

        self.scheduler = NoiseScheduler(num_timesteps=num_timesteps, device=device)
        self.denoiser = UNetWithFiLM(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            base_channels=base_channels,
            num_classes=num_classes,
            device=device,
        )

    def set_timesteps(self, num_timesteps: int) -> None:
        """Update the scheduler's number of timesteps."""
        self.num_timesteps = num_timesteps
        self.scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            device=next(self.parameters()).device,
        )

    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        cfg_dropout: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass with Classifier-Free Guidance (CFG) conditioning dropout.

        Args:
            x0: Clean image batch [B, C, H, W]
            labels: Class labels [B] (values in range [0, num_classes-1])
            cfg_dropout: Probability of replacing label with null token for CFG training.

        Returns:
            Tuple of (loss, pred_noise, true_noise, noisy_x)
        """
        batch_size = x0.size(0)
        device = x0.device

        # 1. Sample random timesteps uniformly
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # 2. Add forward Gaussian noise
        xt, noise = self.scheduler.add_noise(x0, t)

        # 3. Apply CFG conditioning dropout
        if cfg_dropout > 0.0:
            mask = torch.rand(batch_size, device=device) < cfg_dropout
            conditioned_labels = torch.where(
                mask,
                torch.full_like(labels, self.null_token),
                labels,
            )
        else:
            conditioned_labels = labels

        # 4. Predict noise using FiLM-conditioned UNet
        pred_noise = self.denoiser(xt, t, conditioned_labels)

        # 5. MSE Loss
        loss = F.mse_loss(pred_noise, noise)

        return loss, pred_noise, noise, xt

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        shape: Tuple[int, ...],
        device: Optional[Union[str, torch.device]] = None,
        guidance_scale: float = 7.5,
        method: str = "ddim",
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generate images via reverse diffusion with Classifier-Free Guidance (CFG).

        Args:
            labels: Class labels tensor [batch_size].
            shape: Output tensor shape (batch_size, channels, H, W).
            device: Target execution device. If None, uses model parameter device.
            guidance_scale: CFG scale w (1.0 = unconditional/standard conditional, >1.0 = amplified guidance).
            method: Sampling algorithm, 'ddpm' or 'ddim' (default: 'ddim').
            num_inference_steps: Sampling steps. If None, defaults to full training timesteps (e.g. 1000 for DDPM, 50 for DDIM).
            eta: DDIM stochasticity parameter (0.0 = deterministic DDIM, 1.0 = stochastic DDPM-equivalent).
            verbose: Whether to log sampling progress.

        Returns:
            Generated image tensor in range [-1, 1] [batch_size, channels, H, W].
        """
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device) if isinstance(device, str) else device

        if labels.max() >= self.num_classes:
            raise ValueError(f"Label indices must be in [0, {self.num_classes - 1}]")

        if method not in ["ddpm", "ddim"]:
            raise ValueError(f"Sampling method must be 'ddpm' or 'ddim', got '{method}'")

        if num_inference_steps is None:
            num_inference_steps = 50 if method == "ddim" else self.num_timesteps

        batch_size = shape[0]
        img = torch.randn(shape, device=device)

        cond_labels = labels.to(device)
        uncond_labels = torch.full((batch_size,), self.null_token, device=device, dtype=torch.long)

        # Timestep schedule determination
        if num_inference_steps >= self.num_timesteps:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            step_size = self.num_timesteps // num_inference_steps
            timesteps = list(range(self.num_timesteps - 1, -1, -step_size))

        if verbose:
            print(f"Generating {batch_size} samples via {method.upper()} ({len(timesteps)} steps, CFG={guidance_scale}, eta={eta})")

        for idx, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if guidance_scale == 1.0:
                pred_noise = self.denoiser(img, t_tensor, cond_labels)
            else:
                # Batched or dual forward pass for CFG
                pred_noise_uncond = self.denoiser(img, t_tensor, uncond_labels)
                pred_noise_cond = self.denoiser(img, t_tensor, cond_labels)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

            if method == "ddim":
                t_prev = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
                t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                img = self.scheduler.denoise_step_ddim(img, pred_noise, t_tensor, t_prev=t_prev_tensor, eta=eta)
            else:
                img = self.scheduler.denoise_step_ddpm(img, pred_noise, t_tensor)

        return img

    @torch.no_grad()
    def sample_progressive(
        self,
        labels: torch.Tensor,
        shape: Tuple[int, ...],
        device: Optional[Union[str, torch.device]] = None,
        guidance_scale: float = 7.5,
        method: str = "ddim",
        num_inference_steps: Optional[int] = 50,
        eta: float = 0.0,
        save_every: int = 5,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Progressive generation saving intermediate trajectory frames for visualization/GIF.

        Args:
            labels: Class labels tensor [batch_size].
            shape: Output tensor shape (batch_size, channels, H, W).
            device: Target execution device.
            guidance_scale: CFG scale.
            method: 'ddim' or 'ddpm'.
            num_inference_steps: Number of sampling steps.
            eta: DDIM eta.
            save_every: Frequency (in steps) to capture intermediate states.

        Returns:
            Tuple of (final_generated_images, list_of_intermediate_tensors)
        """
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device) if isinstance(device, str) else device

        if num_inference_steps is None:
            num_inference_steps = 50 if method == "ddim" else self.num_timesteps

        batch_size = shape[0]
        img = torch.randn(shape, device=device)

        cond_labels = labels.to(device)
        uncond_labels = torch.full((batch_size,), self.null_token, device=device, dtype=torch.long)

        if num_inference_steps >= self.num_timesteps:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            step_size = self.num_timesteps // num_inference_steps
            timesteps = list(range(self.num_timesteps - 1, -1, -step_size))

        intermediates = [img.clone()]

        for i, t in enumerate(tqdm(timesteps, desc=f"Progressive {method.upper()} Sampling")):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if guidance_scale == 1.0:
                pred_noise = self.denoiser(img, t_tensor, cond_labels)
            else:
                pred_noise_uncond = self.denoiser(img, t_tensor, uncond_labels)
                pred_noise_cond = self.denoiser(img, t_tensor, cond_labels)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

            if method == "ddim":
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                img = self.scheduler.denoise_step_ddim(img, pred_noise, t_tensor, t_prev=t_prev_tensor, eta=eta)
            else:
                img = self.scheduler.denoise_step_ddpm(img, pred_noise, t_tensor)

            if (i + 1) % save_every == 0 or i == len(timesteps) - 1:
                intermediates.append(img.clone())

        return img, intermediates

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint with weights and optional metadata."""
        state = {
            "model_state": self.state_dict(),
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "time_emb_dim": self.time_emb_dim,
                "text_emb_dim": self.text_emb_dim,
                "base_channels": self.base_channels,
                "num_timesteps": self.num_timesteps,
                "num_classes": self.num_classes,
            },
            **kwargs,
        }
        torch.save(state, path)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: Union[str, torch.device] = "cpu",
        **override_kwargs,
    ) -> Tuple["FiLMDiffusion", Dict]:
        """
        Factory method to instantiate and load FiLMDiffusion model from checkpoint.

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        config.update(override_kwargs)

        model = cls(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            time_emb_dim=config.get("time_emb_dim", 128),
            text_emb_dim=config.get("text_emb_dim", 128),
            base_channels=config.get("base_channels", 64),
            num_timesteps=config.get("num_timesteps", 1000),
            num_classes=config.get("num_classes", 10),
            device=device,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        return model, ckpt
