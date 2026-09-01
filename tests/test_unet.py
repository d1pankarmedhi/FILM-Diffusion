import torch
import pytest
from film_diffusion.models.unet import DownBlockWithFiLM, UNetWithFiLM, UpBlockWithFiLM


def test_down_up_blocks():
    down = DownBlockWithFiLM(in_channels=32, out_channels=64, time_emb_dim=128, text_emb_dim=128)
    up = UpBlockWithFiLM(in_channels=64, out_channels=32, time_emb_dim=128, text_emb_dim=128, skip_channels=64)

    x = torch.randn(2, 32, 32, 32)
    t_emb = torch.randn(2, 128)
    text_emb = torch.randn(2, 128)

    down_out, skip = down(x, t_emb, text_emb)
    assert down_out.shape == (2, 64, 16, 16)
    assert skip.shape == (2, 64, 32, 32)

    up_out = up(down_out, skip, t_emb, text_emb)
    assert up_out.shape == (2, 32, 32, 32)


def test_unet_with_film():
    unet = UNetWithFiLM(
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        text_emb_dim=128,
        base_channels=32,
        num_classes=10,
    )
    x = torch.randn(2, 1, 32, 32)
    t = torch.tensor([10, 50])
    labels = torch.tensor([3, 7])

    out = unet(x, t, labels)
    assert out.shape == (2, 1, 32, 32)
