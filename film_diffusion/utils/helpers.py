"""
General Helper Utilities for Seeding, Device Selection, and Configuration Management.
"""

import os
import random
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Select the optimal device based on availability and preference."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(yaml_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from a YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_config(config: Dict[str, Any], yaml_path: str) -> None:
    """Save configuration dictionary to a YAML file."""
    os.makedirs(os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
