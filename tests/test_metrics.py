import numpy as np
import pytest
import torch
from film_diffusion.evaluation.metrics import (
    calculate_activation_statistics,
    calculate_frechet_distance,
    compute_density_coverage,
    compute_inception_score,
    compute_pairwise_distances,
    compute_precision_recall,
)


def test_calculate_frechet_distance():
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.eye(2)
    mu2 = np.array([0.0, 0.0])
    sigma2 = np.eye(2)

    # Identical distributions should have FID == 0
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert np.isclose(fid, 0.0, atol=1e-5)

    # Shifted distribution
    mu3 = np.array([1.0, 1.0])
    fid_shifted = calculate_frechet_distance(mu1, sigma1, mu3, sigma2)
    assert fid_shifted > 0.0


def test_pairwise_distances():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

    dist = compute_pairwise_distances(x, y)
    assert dist.shape == (2, 2)
    assert np.isclose(dist[0, 0].item(), 0.0)
    assert np.isclose(dist[0, 1].item(), 5.0)


def test_precision_recall_density_coverage():
    # 20 samples of 16-dim feature vectors
    real_feats = torch.randn(20, 16)
    fake_feats = real_feats + torch.randn(20, 16) * 0.1

    precision, recall = compute_precision_recall(real_feats, fake_feats, nearest_k=3)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0

    density, coverage = compute_density_coverage(real_feats, fake_feats, nearest_k=3)
    assert density >= 0.0
    assert 0.0 <= coverage <= 1.0


def test_inception_score_computation():
    # Synthetic logits for 10 classes
    logits = np.random.randn(50, 10)
    is_mean, is_std = compute_inception_score(logits, splits=5)
    assert is_mean >= 1.0
    assert is_std >= 0.0
