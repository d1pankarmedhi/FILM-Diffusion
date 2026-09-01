"""
Training Package for FiLM-Diffusion.
"""

from film_diffusion.training.trainer import Trainer, train_text_conditional_diffusion

__all__ = [
    "Trainer",
    "train_text_conditional_diffusion",
]
