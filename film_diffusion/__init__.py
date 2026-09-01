"""
FiLM-Diffusion: Feature-wise Linear Modulation for Efficient Text-Conditional Diffusion Models.
"""

__version__ = "0.1.0"
__author__ = "Dipankar Medhi"

from film_diffusion.data import FASHION_CLASSES, get_dataset, get_fashion_mnist_dataloader
from film_diffusion.evaluation import (
    InceptionV3FeatureExtractor,
    comprehensive_diffusion_metrics,
    compute_fid,
    compute_inception_score,
    compute_kid,
    compute_precision_recall,
    evaluate_metrics_with_precision_recall,
    get_model_parameters,
    get_model_size,
    print_comparison_table,
    save_metrics_to_json,
)
from film_diffusion.models import (
    DownBlockWithFiLM,
    FiLMDiffusion,
    FiLMLayer,
    LabelEncoder,
    NoiseScheduler,
    PositionalEmbedding,
    ResConvBlockWithFiLM,
    TextEncoder,
    UNetWithFiLM,
    UpBlockWithFiLM,
)
from film_diffusion.training import Trainer, train_text_conditional_diffusion
from film_diffusion.utils import (
    create_diffusion_gif,
    get_device,
    load_config,
    plot_training_curve,
    save_config,
    save_samples_grid,
    set_seed,
    show_samples,
    test_specific_generation,
)

__all__ = [
    "__version__",
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
    "comprehensive_diffusion_metrics",
    "get_model_parameters",
    "get_model_size",
    "print_comparison_table",
    "save_metrics_to_json",
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
