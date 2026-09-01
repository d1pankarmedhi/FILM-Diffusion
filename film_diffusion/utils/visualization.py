"""
Visualization and Media Generation Utilities for Diffusion Models.
Includes sample grids, animated GIF rendering, training curves, and test prompts.
"""

import os
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from film_diffusion.data.dataset import FASHION_CLASSES


def show_samples(
    samples: torch.Tensor,
    epoch: Union[int, str] = "",
    labels: Optional[Union[torch.Tensor, List[int]]] = None,
    title_prefix: str = "Samples",
    save_path: Optional[str] = None,
    display: bool = True,
) -> None:
    """
    Plot generated image samples in a grid with optional class subtitles.

    Args:
        samples: Generated image batch tensor [B, C, H, W] in range [-1, 1].
        epoch: Current epoch or step label.
        labels: Class label tensor or list of ints.
        title_prefix: Plot title header.
        save_path: Optional file path to save the generated figure.
        display: Whether to call plt.show().
    """
    # Scale from [-1, 1] to [0, 1]
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    samples = samples.detach().cpu()

    num_samples = min(16, len(samples))
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    if num_samples <= 8:
        fig, axes = plt.subplots(1, num_samples, figsize=(max(2 * num_samples, 4), 2.5))
        axes = [axes] if num_samples == 1 else list(axes)
    else:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(2.5 * grid_size, 2.5 * grid_size))
        axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_samples:
            img = samples[i].squeeze()
            if img.dim() == 3 and img.shape[0] == 1:
                img = img.squeeze(0)

            cmap = "gray" if img.dim() == 2 else None
            axes[i].imshow(img.numpy(), cmap=cmap)

            if labels is not None:
                lbl = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                lbl_name = FASHION_CLASSES.get(lbl, f"Class {lbl}")
                axes[i].set_title(lbl_name, fontsize=9)

        axes[i].axis("off")

    epoch_str = f" at Epoch {epoch}" if epoch != "" else ""
    plt.suptitle(f"{title_prefix}{epoch_str}", fontsize=12)
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if display:
        plt.show()

    plt.close(fig)


def save_samples_grid(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
) -> None:
    """Save raw batch of samples as a clean tiled image grid."""
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    grid = make_grid(samples, nrow=nrow, padding=2)

    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)

    plt.imsave(save_path, grid_np, cmap="gray" if grid_np.ndim == 2 else None)


def create_diffusion_gif(
    intermediates: List[torch.Tensor],
    save_path: str = "diffusion_trajectory.gif",
    duration: int = 60,
    nrow: Optional[int] = None,
) -> None:
    """
    Compile a sequence of progressive sampling intermediate batches into an animated GIF.

    Args:
        intermediates: List of image tensors [B, C, H, W] from progressive sampling.
        save_path: Destination path for the .gif file.
        duration: Frame duration in milliseconds (default: 60ms).
        nrow: Number of items per row in the grid. Defaults to batch size.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    gif_frames = []

    for img_batch in intermediates:
        batch_size = img_batch.size(0)
        row_count = nrow if nrow is not None else batch_size

        grid_tensor = make_grid((img_batch.clamp(-1.0, 1.0) + 1.0) / 2.0, nrow=row_count)
        grid_np = grid_tensor.permute(1, 2, 0).cpu().numpy()

        if grid_np.shape[2] == 1:
            # Duplicate grayscale channel to 3 channels for standard GIF encoding
            grid_np = np.repeat(grid_np, 3, axis=2)

        grid_uint8 = (grid_np * 255.0).astype(np.uint8)
        gif_frames.append(grid_uint8)

    imageio.mimsave(save_path, gif_frames, duration=duration, loop=0)
    print(f"🎬 Animation saved as: {save_path}")


def plot_training_curve(
    training_losses: List[float],
    save_path: Optional[str] = "./training_curve.png",
    display: bool = False,
) -> None:
    """Plot and save training loss convergence curve."""
    plt.figure(figsize=(9, 5))
    plt.plot(
        range(1, len(training_losses) + 1),
        training_losses,
        linewidth=2,
        marker="o",
        markersize=3,
        color="#2980b9",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average MSE Loss", fontsize=12)
    plt.title("FiLM-Conditional Diffusion Training Loss", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"📊 Training curve saved to: {save_path}")

    if display:
        plt.show()

    plt.close()


def test_specific_generation(
    model,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    guidance_scale: float = 7.5,
    save_dir: str = "./test_samples",
    method: str = "ddim",
    steps: int = 50,
) -> None:
    """Run interactive generation across predefined prompt groups."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    examples = [
        ("4 Sneakers", [7, 7, 7, 7]),
        ("Mixed: Dress, Coat, Bag, Boot", [3, 4, 8, 9]),
        ("Complete Wardrobe", list(range(10))),
    ]

    print("\n" + "=" * 60)
    print("🎨 Running Demonstration Generations")
    print("=" * 60)

    for description, label_list in examples:
        labels = torch.tensor(label_list, device=device)
        with torch.no_grad():
            samples = model.sample(
                labels=labels,
                shape=(len(label_list), model.in_channels, 32, 32),
                device=device,
                guidance_scale=guidance_scale,
                method=method,
                num_inference_steps=steps,
            )

        clean_name = description.replace(" ", "_").replace(":", "").replace(",", "")
        save_path = os.path.join(save_dir, f"{clean_name}.png")
        show_samples(samples, epoch="Test", labels=labels, title_prefix=description, save_path=save_path, display=False)
        print(f"  ✓ Generated: {description} -> {save_path}")
