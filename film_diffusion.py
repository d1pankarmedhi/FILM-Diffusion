"""
FiLM-Diffusion: Feature-wise Linear Modulation for Text-Conditional Diffusion Models.

NOTE: This codebase has been refactored into a modular Python package `film_diffusion`.
All modules, layers, trainers, evaluation metrics, and CLI scripts are accessible via:
    import film_diffusion
    from film_diffusion import FiLMDiffusion, Trainer, get_fashion_mnist_dataloader

This file is maintained for backward compatibility.
"""

import sys
from film_diffusion import (
    FASHION_CLASSES,
    DownBlockWithFiLM,
    FiLMDiffusion,
    FiLMLayer,
    InceptionV3FeatureExtractor,
    LabelEncoder,
    NoiseScheduler,
    PositionalEmbedding,
    ResConvBlockWithFiLM,
    TextEncoder,
    Trainer,
    UNetWithFiLM,
    UpBlockWithFiLM,
    calculate_frechet_distance,
    comprehensive_diffusion_metrics,
    compute_density_coverage,
    compute_fid,
    compute_inception_score,
    compute_kid,
    compute_nearest_neighbor_distances,
    compute_pairwise_distances,
    compute_precision_recall,
    create_diffusion_gif,
    evaluate_metrics_with_precision_recall,
    extract_features_from_dataloader,
    extract_features_from_generator,
    get_dataset,
    get_device,
    get_fashion_mnist_dataloader,
    get_model_parameters,
    get_model_size,
    load_config,
    measure_inference_time,
    measure_memory_usage,
    measure_per_timestep_cost,
    measure_single_step_flops,
    plot_training_curve,
    print_comparison_table,
    save_config,
    save_metrics_to_json,
    save_samples_grid,
    set_seed,
    show_samples,
    test_specific_generation,
    train_text_conditional_diffusion,
)

__all__ = [
    "FiLMDiffusion",
    "UNetWithFiLM",
    "NoiseScheduler",
    "FiLMLayer",
    "ResConvBlockWithFiLM",
    "DownBlockWithFiLM",
    "UpBlockWithFiLM",
    "PositionalEmbedding",
    "TextEncoder",
    "LabelEncoder",
    "Trainer",
    "train_text_conditional_diffusion",
    "get_fashion_mnist_dataloader",
    "get_dataset",
    "FASHION_CLASSES",
    "InceptionV3FeatureExtractor",
    "evaluate_metrics_with_precision_recall",
    "compute_fid",
    "compute_inception_score",
    "compute_kid",
    "compute_precision_recall",
    "compute_density_coverage",
    "comprehensive_diffusion_metrics",
    "show_samples",
    "save_samples_grid",
    "create_diffusion_gif",
    "plot_training_curve",
    "test_specific_generation",
    "set_seed",
    "get_device",
    "load_config",
    "save_config",
]

if __name__ == "__main__":
    import quickstart
    quickstart.main()
