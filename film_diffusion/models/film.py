"""
Feature-wise Linear Modulation (FiLM) and FiLM-Conditioned Residual Convolutional Blocks.
Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
"""

from typing import Optional
import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies feature-wise affine transformation conditioned on an external representation (e.g. text/label):
        FiLM(x | c) = gamma(c) * x + beta(c)

    Where:
        - x is the intermediate visual feature map [B, C, H, W]
        - c is the conditioning vector [B, cond_dim]
        - gamma(c) is the channel-wise scale factor [B, C, 1, 1]
        - beta(c) is the channel-wise shift factor [B, C, 1, 1]

    Args:
        num_features (int): Number of feature map channels C.
        cond_dim (int): Dimensionality of the conditioning vector c.
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim

        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)

        # Initialize gamma close to 1 and beta close to 0 for stable initial training dynamics
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature maps [B, C, H, W]
            cond: Conditioning vector [B, cond_dim]

        Returns:
            Modulated feature maps [B, C, H, W]
        """
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return gamma * x + beta


class ResConvBlockWithFiLM(nn.Module):
    """
    Residual Convolutional Block with Timestep Projection and FiLM Conditioning.

    Architecture:
        x -> Conv1 -> GroupNorm1 -> (+ Time Embedding) -> SiLU -> FiLM(cond) -> Conv2 -> GroupNorm2 -> SiLU -> (+ Residual)

    Args:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimension.
        time_emb_dim (int): Timestep embedding dimension.
        cond_dim (int): Text/class conditioning embedding dimension.
        num_groups (int): Number of groups for GroupNorm (default: 8).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_dim: int,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Timestep embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)

        # FiLM layer for conditioning
        self.film = FiLMLayer(out_channels, cond_dim)

        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        # Residual shortcut connection
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input feature maps [B, in_channels, H, W]
            time_emb: Projected timestep embeddings [B, time_emb_dim]
            cond_emb: Text/label conditioning embeddings [B, cond_dim]

        Returns:
            Output feature maps [B, out_channels, H, W]
        """
        residual = self.res_conv(x)

        # 1. First conv + GroupNorm + Time conditioning + Activation
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)

        # 2. FiLM Modulation
        h = self.film(h, cond_emb)

        # 3. Second conv + GroupNorm + Activation
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + residual
