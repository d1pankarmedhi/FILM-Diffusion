import torch
import pytest
from film_diffusion.models.film import FiLMLayer, ResConvBlockWithFiLM


def test_film_layer():
    film = FiLMLayer(num_features=64, cond_dim=128)
    x = torch.randn(4, 64, 16, 16)
    cond = torch.randn(4, 128)
    out = film(x, cond)
    assert out.shape == (4, 64, 16, 16)


def test_res_conv_block_with_film():
    block = ResConvBlockWithFiLM(in_channels=32, out_channels=64, time_emb_dim=128, cond_dim=128)
    x = torch.randn(4, 32, 16, 16)
    t_emb = torch.randn(4, 128)
    cond_emb = torch.randn(4, 128)

    out = block(x, t_emb, cond_emb)
    assert out.shape == (4, 64, 16, 16)
