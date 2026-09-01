import torch
import pytest
from film_diffusion.models.scheduler import NoiseScheduler


def test_scheduler_init():
    scheduler = NoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    assert scheduler.betas.shape == (1000,)
    assert scheduler.alphas.shape == (1000,)
    assert scheduler.alphas_cumprod.shape == (1000,)
    assert scheduler.alphas_cumprod_prev.shape == (1000,)
    assert scheduler.sqrt_alphas_cumprod.shape == (1000,)
    assert scheduler.sqrt_one_minus_alphas_cumprod.shape == (1000,)


def test_scheduler_add_noise():
    scheduler = NoiseScheduler(num_timesteps=100)
    x0 = torch.zeros(4, 1, 32, 32)
    t = torch.tensor([0, 25, 50, 99])

    xt, noise = scheduler.add_noise(x0, t)
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape
    # At t=0, noise weight sqrt(1 - alpha_0) is small
    assert torch.mean(torch.abs(xt[0])) < torch.mean(torch.abs(xt[3]))


def test_scheduler_ddpm_step():
    scheduler = NoiseScheduler(num_timesteps=100)
    xt = torch.randn(4, 1, 32, 32)
    pred_noise = torch.randn(4, 1, 32, 32)
    t = torch.tensor([50, 50, 50, 50])

    denoised = scheduler.denoise_step_ddpm(xt, pred_noise, t)
    assert denoised.shape == xt.shape


def test_scheduler_ddim_deterministic():
    scheduler = NoiseScheduler(num_timesteps=100)
    xt = torch.randn(2, 1, 32, 32)
    pred_noise = torch.randn(2, 1, 32, 32)
    t = torch.tensor([50, 50])
    t_prev = torch.tensor([40, 40])

    # DDIM with eta=0 is deterministic
    out1 = scheduler.denoise_step_ddim(xt, pred_noise, t, t_prev=t_prev, eta=0.0)
    out2 = scheduler.denoise_step_ddim(xt, pred_noise, t, t_prev=t_prev, eta=0.0)
    assert torch.allclose(out1, out2)
