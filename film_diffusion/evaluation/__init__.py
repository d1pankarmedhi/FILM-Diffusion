"""
Evaluation Package for FiLM-Diffusion.
"""

from film_diffusion.evaluation.inception import InceptionV3FeatureExtractor
from film_diffusion.evaluation.metrics import (
    calculate_frechet_distance,
    compute_density_coverage,
    compute_fid,
    compute_inception_score,
    compute_kid,
    compute_nearest_neighbor_distances,
    compute_pairwise_distances,
    compute_precision_recall,
    evaluate_metrics_with_precision_recall,
    extract_features_from_dataloader,
    extract_features_from_generator,
)
from film_diffusion.evaluation.profiler import (
    comprehensive_diffusion_metrics,
    get_model_parameters,
    get_model_size,
    measure_inference_time,
    measure_memory_usage,
    measure_per_timestep_cost,
    measure_single_step_flops,
    print_comparison_table,
    save_metrics_to_json,
)

__all__ = [
    "InceptionV3FeatureExtractor",
    "calculate_frechet_distance",
    "compute_fid",
    "compute_inception_score",
    "compute_kid",
    "compute_pairwise_distances",
    "compute_nearest_neighbor_distances",
    "compute_precision_recall",
    "compute_density_coverage",
    "extract_features_from_dataloader",
    "extract_features_from_generator",
    "evaluate_metrics_with_precision_recall",
    "get_model_parameters",
    "get_model_size",
    "measure_single_step_flops",
    "measure_inference_time",
    "measure_memory_usage",
    "measure_per_timestep_cost",
    "comprehensive_diffusion_metrics",
    "print_comparison_table",
    "save_metrics_to_json",
]
