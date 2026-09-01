"""
Noise Scheduler Module for Diffusion Models.
Supports both DDPM (Ho et al., 2020) and DDIM (Song et al., 2020) sampling schedules.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class NoiseScheduler:
    """
    Gaussian Noise Scheduler for forward and reverse diffusion processes.

    Computes forward noise schedules (beta, alpha, cumulative alpha) and provides
    both standard stochastic DDPM denoising steps and flexible/deterministic DDIM denoising steps.

    Args:
        num_timesteps (int): Total number of diffusion timesteps (default: 1000).
        beta_start (float): Minimum variance schedule value (default: 0.0001).
        beta_end (float): Maximum variance schedule value (default: 0.02).
        schedule (str): Schedule type, 'linear' or 'cosine' (default: 'linear').
        device (str or torch.device): Device on which precalculated schedules reside.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: Union[str, torch.device] = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = torch.device(device) if isinstance(device, str) else device

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=self.device)
        elif schedule == "cosine":
            # Cosine schedule from Improved Denoising Diffusion (Nichol & Dhariwal, 2021)
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32, device=self.device)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, min=beta_start, max=beta_end)
        else:
            raise ValueError(f"Unsupported schedule '{schedule}'. Choose 'linear' or 'cosine'.")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device, dtype=torch.float32), self.alphas_cumprod[:-1]]
        )

        # Precalculated terms for forward diffusion (q(x_t | x_0))
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: Union[str, torch.device]) -> "NoiseScheduler":
        """Move scheduler schedule tensors to target device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        return self

    def add_noise(
        self,
        original_image: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0).

        Args:
            original_image: Clean input images [B, C, H, W]
            t: Timesteps for each sample in batch [B]
            noise: Optional Gaussian noise [B, C, H, W]. Sampled if None.

        Returns:
            Tuple of (noisy_image x_t, sampled noise epsilon)
        """
        device = original_image.device
        if noise is None:
            noise = torch.randn_like(original_image)

        t_long = t.long().to(self.device)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_long].reshape(-1, 1, 1, 1).to(device)
        sqrt_one_minus_alpha_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod[t_long].reshape(-1, 1, 1, 1).to(device)
        )

        noisy_image = sqrt_alpha_cumprod_t * original_image + sqrt_one_minus_alpha_cumprod_t * noise
        return noisy_image, noise

    def denoise_step_ddpm(
        self,
        noisy_image: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDPM reverse diffusion step: p_theta(x_{t-1} | x_t).

        Args:
            noisy_image: Current noisy image x_t [B, C, H, W]
            predicted_noise: Model's predicted noise epsilon_theta(x_t, t) [B, C, H, W]
            t: Current timesteps [B]

        Returns:
            Denoised image estimate x_{t-1} [B, C, H, W]
        """
        device = noisy_image.device
        t_long = t.long().to(self.device)

        alpha_t = self.alphas[t_long].reshape(-1, 1, 1, 1).to(device)
        alpha_cumprod_t = self.alphas_cumprod[t_long].reshape(-1, 1, 1, 1).to(device)
        beta_t = self.betas[t_long].reshape(-1, 1, 1, 1).to(device)

        model_mean = (1.0 / torch.sqrt(alpha_t)) * (
            noisy_image - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
        )

        if (t_long > 0).any():
            noise = torch.randn_like(noisy_image)
            model_variance = beta_t
            return model_mean + torch.sqrt(model_variance) * noise
        else:
            return model_mean

    def denoise_step_ddim(
        self,
        noisy_image: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
        t_prev: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single DDIM reverse diffusion step (Song et al., 2020).

        Allows deterministic (eta=0.0) or stochastic (0 < eta <= 1.0) sampling
        across arbitrary sub-sampled timestep trajectories.

        Args:
            noisy_image: Current noisy image x_t [B, C, H, W]
            predicted_noise: Model's predicted noise epsilon_theta(x_t, t) [B, C, H, W]
            t: Current timesteps [B]
            t_prev: Previous timesteps [B]. If None, defaults to t - 1.
            eta: Stochasticity parameter (0.0 = deterministic DDIM, 1.0 = DDPM-like).

        Returns:
            Denoised image estimate x_{t_prev} [B, C, H, W]
        """
        device = noisy_image.device
        t_long = t.long().to(self.device)

        alpha_cumprod_t = self.alphas_cumprod[t_long].reshape(-1, 1, 1, 1).to(device)

        if t_prev is None:
            alpha_cumprod_t_prev = self.alphas_cumprod_prev[t_long].reshape(-1, 1, 1, 1).to(device)
        else:
            t_prev_long = t_prev.long().to(self.device)
            alpha_cumprod_t_prev = torch.where(
                t_prev_long >= 0,
                self.alphas_cumprod[t_prev_long.clamp(min=0)],
                torch.tensor(1.0, device=self.device),
            ).reshape(-1, 1, 1, 1).to(device)

        # 1. Predict x_0 from x_t and predicted noise
        pred_x0 = (noisy_image - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 2. Calculate variance for stochastic term
        if eta > 0.0:
            variance = (
                eta
                * torch.sqrt((1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t))
                * torch.sqrt(1.0 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_t_prev - variance**2, min=0.0)) * predicted_noise
        else:
            variance = torch.zeros_like(alpha_cumprod_t)
            dir_xt = torch.sqrt(1.0 - alpha_cumprod_t_prev) * predicted_noise

        # 3. Compute x_{t-1}
        denoised_image = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt

        if eta > 0.0 and (t_long > 0).any():
            noise = torch.randn_like(noisy_image)
            denoised_image = denoised_image + variance * noise

        return denoised_image

    # Backward compatibility alias
    _denoise_step_ddim = denoise_step_ddim
