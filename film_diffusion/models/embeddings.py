"""
Embedding modules for Diffusion Timestep and Text/Class Label Conditioning.
"""

from typing import Optional, Union
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding for diffusion timesteps.

    Maps scalar/1D timestep tensor t to continuous sinusoidal embedding vectors:
    [sin(t * omega), cos(t * omega)].

    Args:
        embedding_dim (int): Total dimensionality of the embedding (must be even).
        device (str or torch.device): Initial device for div_term buffer.
    """

    def __init__(self, embedding_dim: int, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")

        self.embedding_dim = embedding_dim
        self.half_embedding_dim = embedding_dim // 2

        # Precompute frequency division term
        div_term = torch.exp(
            torch.arange(0, self.half_embedding_dim, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / self.half_embedding_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape [batch_size] or [batch_size, 1]

        Returns:
            Embeddings of shape [batch_size, embedding_dim]
        """
        if t.dim() > 1:
            t = t.view(-1)

        t = t.to(self.div_term.device).float()
        emb = t[:, None] * self.div_term[None, :]  # [batch, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [batch, embedding_dim]
        return emb


class TextEncoder(nn.Module):
    """
    Embedding-based Label/Text Encoder for categorical class conditioning.
    Maps discrete class labels (0 to num_classes-1) or null token (num_classes)
    to continuous conditioning vectors through embedding lookup and MLP projection.

    Args:
        num_classes (int): Number of distinct conditioning classes (default: 10).
        embed_dim (int): Dimensionality of the output conditioning vector (default: 128).
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # num_classes + 1 to accommodate the null token used in Classifier-Free Guidance (CFG)
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Integer tensor of class labels [batch_size] in range [0, num_classes].

        Returns:
            Conditioning embedding tensor of shape [batch_size, embed_dim].
        """
        if labels.max() > self.num_classes:
            raise ValueError(
                f"Label index {labels.max().item()} exceeds maximum valid index {self.num_classes}"
            )
        return self.mlp(self.embedding(labels))


# Backward compatibility alias
LabelEncoder = TextEncoder
