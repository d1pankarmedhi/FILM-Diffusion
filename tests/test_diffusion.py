import torch
import pytest
from film_diffusion.models.diffusion import FiLMDiffusion


def test_film_diffusion_forward():
    model = FiLMDiffusion(base_channels=32, num_timesteps=100)
    x0 = torch.randn(2, 1, 32, 32)
    labels = torch.tensor([1, 8])

    loss, pred_noise, true_noise, xt = model(x0, labels, cfg_dropout=0.1)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0
    assert pred_noise.shape == x0.shape
    assert true_noise.shape == x0.shape
    assert xt.shape == x0.shape


def test_film_diffusion_sample_ddim():
    model = FiLMDiffusion(base_channels=32, num_timesteps=100)
    labels = torch.tensor([0, 4])
    shape = (2, 1, 32, 32)

    samples = model.sample(
        labels=labels,
        shape=shape,
        guidance_scale=7.5,
        method="ddim",
        num_inference_steps=10,
        eta=0.0,
    )
    assert samples.shape == shape


def test_film_diffusion_sample_progressive():
    model = FiLMDiffusion(base_channels=32, num_timesteps=100)
    labels = torch.tensor([2, 5])
    shape = (2, 1, 32, 32)

    final_samples, intermediates = model.sample_progressive(
        labels=labels,
        shape=shape,
        guidance_scale=7.5,
        method="ddim",
        num_inference_steps=10,
        save_every=5,
    )
    assert final_samples.shape == shape
    assert len(intermediates) >= 2
