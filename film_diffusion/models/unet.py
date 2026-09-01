"""
U-Net Backbone with FiLM Conditioning for Text/Class-Conditional Diffusion Models.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from film_diffusion.models.embeddings import PositionalEmbedding, TextEncoder
from film_diffusion.models.film import FiLMLayer, ResConvBlockWithFiLM


class DownBlockWithFiLM(nn.Module):
    """
    Downsampling block with FiLM conditioning.
    Combines a ResConvBlockWithFiLM with strided 2D convolution for downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        text_emb_dim: int,
    ):
        super().__init__()
        self.res_conv = ResConvBlockWithFiLM(in_channels, out_channels, time_emb_dim, text_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            downsampled_x: [B, out_channels, H/2, W/2]
            skip_connection: [B, out_channels, H, W]
        """
        x = self.res_conv(x, t_emb, text_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlockWithFiLM(nn.Module):
    """
    Upsampling block with FiLM conditioning.
    Combines ConvTranspose2d upsampling with skip-connection concatenation and ResConvBlockWithFiLM.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        text_emb_dim: int,
        skip_channels: int,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res_conv = ResConvBlockWithFiLM(
            in_channels + skip_channels, out_channels, time_emb_dim, text_emb_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Feature map from previous level [B, in_channels, H, W]
            skip: Skip connection feature map [B, skip_channels, 2*H, 2*W]
            t_emb: Timestep embedding [B, time_emb_dim]
            text_emb: Conditioning embedding [B, text_emb_dim]

        Returns:
            upsampled_x: [B, out_channels, 2*H, 2*W]
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_conv(x, t_emb, text_emb)
        return x


class UNetWithFiLM(nn.Module):
    """
    U-Net Denoiser with Feature-wise Linear Modulation (FiLM) conditioning.

    Replaces standard cross-attention layers with FiLM modulation throughout
    the encoder, bottleneck, and decoder stages.

    Args:
        in_channels (int): Number of image input channels (default: 1 for grayscale / FashionMNIST).
        out_channels (int): Number of image output channels (default: 1).
        time_emb_dim (int): Dimensionality of timestep embedding (default: 128).
        text_emb_dim (int): Dimensionality of text/label conditioning embedding (default: 128).
        base_channels (int): Base channel multiplier (default: 64; supports 32, 64, 96, 128).
        num_classes (int): Number of conditioning classes (default: 10).
        device (str or torch.device): Initial device.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_emb_dim: int = 128,
        text_emb_dim: int = 128,
        base_channels: int = 64,
        num_classes: int = 10,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim
        self.base_channels = base_channels
        self.num_classes = num_classes

        # 1. Timestep Embedding Network
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim, device=device),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 2. Text / Class Label Encoder
        self.text_encoder = TextEncoder(num_classes=num_classes, embed_dim=text_emb_dim)

        # 3. Initial Convolution
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 4. Encoder (Downsampling Path)
        self.down1 = DownBlockWithFiLM(
            base_channels, base_channels * 2, time_emb_dim, text_emb_dim
        )
        self.down2 = DownBlockWithFiLM(
            base_channels * 2, base_channels * 4, time_emb_dim, text_emb_dim
        )

        # 5. Bottleneck
        self.bottleneck = ResConvBlockWithFiLM(
            base_channels * 4, base_channels * 4, time_emb_dim, text_emb_dim
        )

        # 6. Decoder (Upsampling Path)
        self.up1 = UpBlockWithFiLM(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            skip_channels=base_channels * 4,
        )
        self.up2 = UpBlockWithFiLM(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            skip_channels=base_channels * 2,
        )

        # 7. Final Output Convolution
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward denoising pass.

        Args:
            x: Noisy image batch [B, in_channels, H, W]
            t: Timesteps batch [B]
            labels: Conditioning class labels [B]

        Returns:
            Predicted noise tensor [B, out_channels, H, W]
        """
        # Embed timesteps and labels
        t_emb = self.time_mlp(t)
        text_emb = self.text_encoder(labels)

        # Encoder
        x = self.initial_conv(x)
        x1, skip1 = self.down1(x, t_emb, text_emb)
        x2, skip2 = self.down2(x1, t_emb, text_emb)

        # Bottleneck
        x3 = self.bottleneck(x2, t_emb, text_emb)

        # Decoder
        x = self.up1(x3, skip2, t_emb, text_emb)
        x = self.up2(x, skip1, t_emb, text_emb)

        return self.final_conv(x)
