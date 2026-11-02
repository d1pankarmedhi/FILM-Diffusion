"""
Hardware and Computational Efficiency Profiler for Diffusion Models.
Measures parameters, FLOPs/MACs, latency, throughput, and peak memory consumption.
"""

import json
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import psutil
import torch
import torch.nn as nn

# Graceful optional imports for FLOPs profilers
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


def get_model_parameters(model: nn.Module) -> Dict[str, float]:
    """Calculate total, trainable, and non-trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_params": int(total_params),
        "total_params_M": float(total_params / 1e6),
        "trainable_params": int(trainable_params),
        "trainable_params_M": float(trainable_params / 1e6),
        "non_trainable_params": int(non_trainable_params),
        "non_trainable_params_M": float(non_trainable_params / 1e6),
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Calculate model parameter and buffer memory size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    return {
        "param_size_mb": float(param_size / (1024**2)),
        "buffer_size_mb": float(buffer_size / (1024**2)),
        "total_size_mb": float(total_size / (1024**2)),
    }


def measure_single_step_flops(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = "fvcore",
) -> Dict[str, Any]:
    """
    Measure floating point operations (FLOPs) and Multiply-Accumulate operations (MACs)
    for a single forward denoising step.
    """
    model.eval()
    denoiser = getattr(model, "denoiser", model)
    results: Dict[str, Any] = {}

    if method == "fvcore" and FVCORE_AVAILABLE:
        inputs = (x_t, t, labels) if labels is not None else (x_t, t)
        flops = FlopCountAnalysis(denoiser, inputs)
        results["flops"] = int(flops.total())
        results["flops_G"] = float(flops.total() / 1e9)
        results["flops_by_operator"] = dict(flops.by_operator())
    elif method == "thop" and THOP_AVAILABLE:
        inputs = (x_t, t, labels) if labels is not None else (x_t, t)
        macs, params = thop_profile(denoiser, inputs=inputs, verbose=False)
        results["macs"] = int(macs)
        results["macs_G"] = float(macs / 1e9)
        results["flops"] = int(2 * macs)
        results["flops_G"] = float((2 * macs) / 1e9)
        results["params"] = int(params)
    elif method == "ptflops" and PTFLOPS_AVAILABLE:
        try:
            input_shape = tuple(x_t.shape[1:])
            macs, params = get_model_complexity_info(
                denoiser, input_shape, as_strings=False, print_per_layer_stat=False
            )
            results["macs"] = int(macs)
            results["macs_G"] = float(macs / 1e9)
            results["flops"] = int(2 * macs)
            results["flops_G"] = float((2 * macs) / 1e9)
            results["params"] = int(params)
        except Exception as e:
            results["error"] = str(e)
    else:
        results["warning"] = f"Profiler method '{method}' is not available. Install fvcore/thop/ptflops."

    return results


def measure_inference_time(
    model: nn.Module,
    sample_fn: Callable,
    labels: torch.Tensor,
    shape: Tuple[int, ...],
    device: torch.device,
    guidance_scale: float = 7.5,
    num_runs: int = 10,
    warmup_runs: int = 2,
) -> Dict[str, float]:
    """Measure inference latency and throughput over multiple runs with warmup and CUDA synchronization."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = sample_fn(labels=labels, shape=shape, device=device, guidance_scale=guidance_scale)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = sample_fn(labels=labels, shape=shape, device=device, guidance_scale=guidance_scale)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

    times_arr = np.array(times)
    batch_size = shape[0]

    return {
        "mean_time_s": float(np.mean(times_arr)),
        "std_time_s": float(np.std(times_arr)),
        "min_time_s": float(np.min(times_arr)),
        "max_time_s": float(np.max(times_arr)),
        "median_time_s": float(np.median(times_arr)),
        "samples_per_second": float(batch_size / np.mean(times_arr)),
        "time_per_sample_s": float(np.mean(times_arr) / batch_size),
    }


def measure_memory_usage(
    model: nn.Module,
    sample_fn: Callable,
    labels: torch.Tensor,
    shape: Tuple[int, ...],
    device: torch.device,
    guidance_scale: float = 7.5,
) -> Dict[str, float]:
    """Measure peak and allocated memory during sampling."""
    model.eval()
    results = {}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        mem_before = torch.cuda.memory_allocated(device) / (1024**2)

        with torch.no_grad():
            _ = sample_fn(labels=labels, shape=shape, device=device, guidance_scale=guidance_scale)

        mem_after = torch.cuda.memory_allocated(device) / (1024**2)
        mem_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)

        results.update({
            "memory_before_mb": float(mem_before),
            "memory_after_mb": float(mem_after),
            "memory_peak_mb": float(mem_peak),
            "memory_reserved_mb": float(mem_reserved),
            "memory_used_mb": float(mem_after - mem_before),
        })
    else:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)

        with torch.no_grad():
            _ = sample_fn(labels=labels, shape=shape, device=device, guidance_scale=guidance_scale)

        mem_after = process.memory_info().rss / (1024**2)
        results.update({
            "memory_before_mb": float(mem_before),
            "memory_after_mb": float(mem_after),
            "memory_used_mb": float(mem_after - mem_before),
        })

    return results


def measure_per_timestep_cost(
    model: nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Measure single timestep evaluation latency."""
    model.eval()
    device = x_t.device
    denoiser = getattr(model, "denoiser", model)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = denoiser(x_t, timesteps, labels)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = denoiser(x_t, timesteps, labels)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    times_arr = np.array(times)
    return {
        "mean_step_time_ms": float(np.mean(times_arr) * 1000),
        "std_step_time_ms": float(np.std(times_arr) * 1000),
        "min_step_time_ms": float(np.min(times_arr) * 1000),
        "max_step_time_ms": float(np.max(times_arr) * 1000),
    }


def comprehensive_diffusion_metrics(
    model: nn.Module,
    sample_fn: Callable,
    labels: torch.Tensor,
    shape: Tuple[int, ...],
    device: torch.device,
    num_timesteps: int = 1000,
    guidance_scale: float = 7.5,
    num_inference_runs: int = 5,
    flops_method: str = "fvcore",
) -> Dict[str, Any]:
    """Run full suite of hardware and compute profiling benchmarks."""
    metrics: Dict[str, Any] = {}

    # 1. Parameters & Size
    metrics["parameters"] = get_model_parameters(model)
    metrics["model_size"] = get_model_size(model)

    # 2. FLOPs
    batch_size = shape[0]
    x_dummy = torch.randn(shape, device=device)
    t_dummy = torch.randint(0, num_timesteps, (batch_size,), device=device)
    num_classes = getattr(model, "num_classes", 10)
    labels_dummy = torch.randint(0, num_classes, (batch_size,), device=device)

    metrics["single_step_flops"] = measure_single_step_flops(
        model, x_dummy, t_dummy, labels_dummy, method=flops_method
    )

    # 3. Timestep latency
    metrics["per_timestep"] = measure_per_timestep_cost(
        model, x_dummy, t_dummy, labels_dummy, num_runs=50
    )

    # 4. Sampling time
    metrics["inference_time"] = measure_inference_time(
        model, sample_fn, labels, shape, device,
        guidance_scale=guidance_scale, num_runs=num_inference_runs
    )

    # 5. Memory
    metrics["memory"] = measure_memory_usage(
        model, sample_fn, labels, shape, device, guidance_scale=guidance_scale
    )

    # 6. Derived metrics
    derived = {
        "latency_ms": metrics["inference_time"]["mean_time_s"] * 1000,
        "throughput_samples_per_sec": metrics["inference_time"]["samples_per_second"],
    }
    if "flops_G" in metrics.get("single_step_flops", {}):
        total_flops_G = metrics["single_step_flops"]["flops_G"] * num_timesteps
        derived["total_sampling_flops_G"] = total_flops_G
        derived["flops_per_second_G"] = total_flops_G / metrics["inference_time"]["mean_time_s"]
        derived["flops_per_sample_G"] = total_flops_G / shape[0]

    metrics["derived"] = derived
    return metrics


def print_comparison_table(metrics_dict: Dict[str, Dict[str, Any]]) -> None:
    """Pretty print a tabular summary comparing multiple model configurations."""
    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 90)

    rows = [
        ("Total Params (M)", lambda m: f"{m['parameters']['total_params_M']:.2f}"),
        ("Model Size (MB)", lambda m: f"{m['model_size']['total_size_mb']:.2f}"),
        ("FLOPs / Step (G)", lambda m: f"{m['single_step_flops'].get('flops_G', 'N/A')}"),
        ("Sampling Time (s)", lambda m: f"{m['inference_time']['mean_time_s']:.3f}"),
        ("Throughput (samples/s)", lambda m: f"{m['inference_time']['samples_per_second']:.2f}"),
        ("Step Time (ms)", lambda m: f"{m['per_timestep']['mean_step_time_ms']:.2f}"),
        ("Peak Memory (MB)", lambda m: f"{m['memory'].get('memory_peak_mb', m['memory'].get('memory_used_mb', 'N/A'))}"),
    ]

    header = f"{'Metric':<25}" + "".join(f"{name:>20}" for name in metrics_dict.keys())
    print(header)
    print("-" * len(header))

    for label, fn in rows:
        row_str = f"{label:<25}"
        for _, m in metrics_dict.items():
            try:
                val = fn(m)
                row_str += f"{str(val):>20}"
            except Exception:
                row_str += f"{'N/A':>20}"
        print(row_str)

    print("=" * 90 + "\n")


def save_metrics_to_json(metrics: Dict, filepath: str) -> None:
    """Save serializable metrics dictionary to JSON file."""
    def serialize_obj(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: serialize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_obj(i) for i in obj]
        return str(obj)

    clean_metrics = serialize_obj(metrics)
    with open(filepath, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"💾 Profiling metrics saved to: {filepath}")
