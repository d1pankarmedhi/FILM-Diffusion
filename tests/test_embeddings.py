import torch
import pytest
from film_diffusion.models.embeddings import PositionalEmbedding, TextEncoder


def test_positional_embedding():
    emb = PositionalEmbedding(embedding_dim=128)
    t = torch.tensor([0, 50, 100, 999])
    out = emb(t)
    assert out.shape == (4, 128)

    # 2D input flattening
    t_2d = torch.tensor([[10], [20]])
    out_2d = emb(t_2d)
    assert out_2d.shape == (2, 128)


def test_text_encoder():
    encoder = TextEncoder(num_classes=10, embed_dim=128)
    labels = torch.tensor([0, 5, 9, 10])  # 10 is the null token
    out = encoder(labels)
    assert out.shape == (4, 128)

    # Test out of bounds label
    with pytest.raises(ValueError):
        encoder(torch.tensor([11]))
