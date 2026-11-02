"""
Quantitative Evaluation Metrics for Generative Image Models.
Implements FID, IS, KID, Improved Precision & Recall (Sajjadi/Kynkäänniemi),
and Density & Coverage (Naeem et al.).
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from film_diffusion.evaluation.inception import InceptionV3FeatureExtractor
from film_diffusion.models.diffusion import FiLMDiffusion


# ============================================================================
# 1. FID (Fréchet Inception Distance)
# ============================================================================

def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Numpy implementation of the Fréchet Distance:
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product may be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(features: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance matrix for a set of feature vectors."""
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(
    real_features: Union[torch.Tensor, np.ndarray],
    fake_features: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute Fréchet Inception Distance between real and fake feature sets."""
    mu_real, sigma_real = calculate_activation_statistics(real_features)
    mu_fake, sigma_fake = calculate_activation_statistics(fake_features)
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


# ============================================================================
# 2. IS (Inception Score)
# ============================================================================

def compute_inception_score(
    logits: Union[torch.Tensor, np.ndarray],
    splits: int = 10,
    eps: float = 1e-16,
) -> Tuple[float, float]:
    """
    Compute Inception Score (mean and standard deviation) across splits.
    IS = exp(E_x[KL(p(y|x) || p(y))])
    """
    if torch.is_tensor(logits):
        logits = logits.cpu().numpy()

    # Convert logits to probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    n_samples = len(probs)
    split_size = n_samples // splits
    scores = []

    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size, :]
        p_y = np.expand_dims(np.mean(part, axis=0), 0)
        kl = part * (np.log(part + eps) - np.log(p_y + eps))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))


# ============================================================================
# 3. KID (Kernel Inception Distance)
# ============================================================================

def compute_kid(
    real_features: Union[torch.Tensor, np.ndarray],
    fake_features: Union[torch.Tensor, np.ndarray],
    num_subsets: int = 100,
    max_subset_size: int = 1000,
) -> Tuple[float, float]:
    """
    Compute Kernel Inception Distance (Bińkowski et al., 2018) using polynomial kernel:
    k(x, y) = ((1/d) * x^T y + 1)^3
    """
    if torch.is_tensor(real_features):
        real_features = real_features.cpu().numpy()
    if torch.is_tensor(fake_features):
        fake_features = fake_features.cpu().numpy()

    n_real = len(real_features)
    n_fake = len(fake_features)
    subset_size = min(n_real, n_fake, max_subset_size)
    d = real_features.shape[1]

    def polynomial_kernel(x, y):
        return ((x.dot(y.T) / d) + 1.0) ** 3

    mmd_list = []
    for _ in range(num_subsets):
        r_idx = np.random.choice(n_real, subset_size, replace=False)
        f_idx = np.random.choice(n_fake, subset_size, replace=False)

        x = real_features[r_idx]
        y = fake_features[f_idx]

        k_xx = polynomial_kernel(x, x)
        k_yy = polynomial_kernel(y, y)
        k_xy = polynomial_kernel(x, y)

        # Unbiased estimator of MMD^2
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        mmd = (
            k_xx.sum() / (subset_size * (subset_size - 1))
            + k_yy.sum() / (subset_size * (subset_size - 1))
            - 2.0 * k_xy.mean()
        )
        mmd_list.append(mmd)

    return float(np.mean(mmd_list) * 1000), float(np.std(mmd_list) * 1000)


# ============================================================================
# 4. Precision, Recall, Density, and Coverage
# ============================================================================

def compute_pairwise_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Efficient batch Euclidean pairwise distance between X and Y."""
    X_norm = (X**2).sum(dim=1).view(-1, 1)
    Y_norm = (Y**2).sum(dim=1).view(1, -1)
    dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
    return torch.clamp(dist, min=0.0).sqrt()


def compute_nearest_neighbor_distances(X: torch.Tensor, Y: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Compute distances from each point in X to its k-th nearest neighbor in Y."""
    dist_matrix = compute_pairwise_distances(X, Y)
    sorted_dist, _ = torch.sort(dist_matrix, dim=1)
    k_index = min(k, sorted_dist.shape[1] - 1)
    return sorted_dist[:, k_index]


def compute_precision_recall(
    real_features: Union[torch.Tensor, np.ndarray],
    fake_features: Union[torch.Tensor, np.ndarray],
    nearest_k: int = 3,
) -> Tuple[float, float]:
    """
    Compute Improved Precision and Recall for Generative Models (Kynkäänniemi et al., NeurIPS 2019).
    Precision: fidelity of generated samples to real data distribution.
    Recall: diversity/coverage of generated samples over real distribution.
    """
    if isinstance(real_features, np.ndarray):
        real_features = torch.from_numpy(real_features).float()
    if isinstance(fake_features, np.ndarray):
        fake_features = torch.from_numpy(fake_features).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_features = real_features.to(device)
    fake_features = fake_features.to(device)

    # 1. Real data hypersphere radii
    real_to_real_k = compute_nearest_neighbor_distances(real_features, real_features, k=nearest_k)
    # 2. Fake data hypersphere radii
    fake_to_fake_k = compute_nearest_neighbor_distances(fake_features, fake_features, k=nearest_k)

    # 3. Precision: Is fake sample within ANY real hypersphere?
    fake_to_real_dist = compute_pairwise_distances(fake_features, real_features)
    precision_mask = (fake_to_real_dist <= real_to_real_k.unsqueeze(0)).any(dim=1)
    precision = precision_mask.float().mean().item()

    # 4. Recall: Is real sample within ANY fake hypersphere?
    real_to_fake_dist = compute_pairwise_distances(real_features, fake_features)
    recall_mask = (real_to_fake_dist <= fake_to_fake_k.unsqueeze(0)).any(dim=1)
    recall = recall_mask.float().mean().item()

    return float(precision), float(recall)


def compute_density_coverage(
    real_features: Union[torch.Tensor, np.ndarray],
    fake_features: Union[torch.Tensor, np.ndarray],
    nearest_k: int = 3,
) -> Tuple[float, float]:
    """Compute Density and Coverage (Naeem et al., ICML 2020)."""
    if isinstance(real_features, np.ndarray):
        real_features = torch.from_numpy(real_features).float()
    if isinstance(fake_features, np.ndarray):
        fake_features = torch.from_numpy(fake_features).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_features = real_features.to(device)
    fake_features = fake_features.to(device)

    real_to_real_k = compute_nearest_neighbor_distances(real_features, real_features, k=nearest_k)
    fake_to_real_dist = compute_pairwise_distances(fake_features, real_features)

    # Density: average count of real spheres containing fake samples / k
    density = (fake_to_real_dist <= real_to_real_k.unsqueeze(0)).float().sum(dim=1).mean().item() / float(nearest_k)

    # Coverage: fraction of real spheres containing at least one fake sample
    coverage = (fake_to_real_dist <= real_to_real_k.unsqueeze(0)).any(dim=0).float().mean().item()

    return float(density), float(coverage)


# ============================================================================
# 5. Feature Extraction Pipelines
# ============================================================================

def extract_features_from_dataloader(
    dataloader: DataLoader,
    feature_extractor: InceptionV3FeatureExtractor,
    max_samples: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract Inception pool3 features and logits from a DataLoader."""
    features_list = []
    logits_list = []
    total_extracted = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting Real Features"):
            images = images.to(device)
            feats, logs = feature_extractor(images)

            features_list.append(feats.cpu())
            logits_list.append(logs.cpu())
            total_extracted += len(images)

            if total_extracted >= max_samples:
                break

    features = torch.cat(features_list, dim=0)[:max_samples]
    logits = torch.cat(logits_list, dim=0)[:max_samples]
    return features, logits


def extract_features_from_generator(
    model: FiLMDiffusion,
    feature_extractor: InceptionV3FeatureExtractor,
    num_samples: int = 10000,
    batch_size: int = 128,
    guidance_scale: float = 7.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    method: str = "ddim",
    num_inference_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate samples and extract Inception features on-the-fly."""
    model.eval()
    features_list = []
    logits_list = []
    samples_generated = 0

    num_classes = model.num_classes
    pbar = tqdm(total=num_samples, desc=f"Generating & Extracting ({method.upper()})")

    with torch.no_grad():
        while samples_generated < num_samples:
            current_batch = min(batch_size, num_samples - samples_generated)
            labels = torch.randint(0, num_classes, (current_batch,), device=device)

            samples = model.sample(
                labels=labels,
                shape=(current_batch, model.in_channels, 32, 32),
                device=device,
                guidance_scale=guidance_scale,
                method=method,
                num_inference_steps=num_inference_steps,
            )

            feats, logs = feature_extractor(samples)
            features_list.append(feats.cpu())
            logits_list.append(logs.cpu())

            samples_generated += current_batch
            pbar.update(current_batch)

    pbar.close()
    features = torch.cat(features_list, dim=0)[:num_samples]
    logits = torch.cat(logits_list, dim=0)[:num_samples]
    return features, logits


# ============================================================================
# 6. Complete Evaluation Orchestration
# ============================================================================

def evaluate_metrics_with_precision_recall(
    model: FiLMDiffusion,
    real_dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_samples: int = 1000,
    guidance_scale: float = 7.5,
    nearest_k: int = 3,
    save_dir: str = "./evaluation",
    method: str = "ddim",
    num_inference_steps: int = 50,
) -> Dict[str, float]:
    """
    Run complete evaluation pipeline: FID, IS, KID, Precision, Recall, Density, Coverage.
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_extractor = InceptionV3FeatureExtractor(device=device)

    print("\n" + "=" * 70)
    print("🔬 COMPREHENSIVE GENERATIVE QUALITY EVALUATION")
    print(f"📊 Samples: {num_samples} | Guidance: {guidance_scale} | Sampler: {method.upper()} ({num_inference_steps} steps)")
    print("=" * 70)

    # 1. Extract Real Features
    real_feats, real_logs = extract_features_from_dataloader(
        real_dataloader, feature_extractor, max_samples=num_samples, device=device
    )

    # 2. Extract Fake Features
    fake_feats, fake_logs = extract_features_from_generator(
        model, feature_extractor, num_samples=num_samples, guidance_scale=guidance_scale,
        device=device, method=method, num_inference_steps=num_inference_steps
    )

    # 3. Compute Metrics
    print("\nComputing Fréchet Inception Distance (FID)...")
    fid = compute_fid(real_feats, fake_feats)

    print("Computing Inception Score (IS)...")
    is_mean, is_std = compute_inception_score(fake_logs)

    print("Computing Kernel Inception Distance (KID)...")
    kid_mean, kid_std = compute_kid(real_feats, fake_feats)

    print("Computing Precision & Recall...")
    precision, recall = compute_precision_recall(real_feats, fake_feats, nearest_k=nearest_k)

    print("Computing Density & Coverage...")
    density, coverage = compute_density_coverage(real_feats, fake_feats, nearest_k=nearest_k)

    results = {
        "fid": float(fid),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
        "guidance_scale": float(guidance_scale),
        "num_samples": int(num_samples),
        "sampler": method,
        "inference_steps": int(num_inference_steps),
    }

    print("\n" + "-" * 40)
    print(f"  FID (↓):                 {fid:.3f}")
    print(f"  Inception Score (↑):     {is_mean:.3f} ± {is_std:.3f}")
    print(f"  KID x 10^-3 (↓):         {kid_mean:.3f} ± {kid_std:.3f}")
    print(f"  Precision (↑):           {precision:.4f}")
    print(f"  Recall (↑):              {recall:.4f}")
    print(f"  Density:                 {density:.4f}")
    print(f"  Coverage:                {coverage:.4f}")
    print("-" * 40)

    # Save to JSON
    json_path = os.path.join(save_dir, "metrics_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Results saved to: {json_path}\n")

    return results
