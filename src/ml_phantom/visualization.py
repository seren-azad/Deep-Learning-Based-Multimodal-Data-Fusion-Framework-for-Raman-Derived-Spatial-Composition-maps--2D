from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

PHASE_COLORS = {
    "B": np.array([1.0, 1.0, 1.0]),
    "C1": np.array([0.9, 0.9, 0.9]),
    "C2": np.array([0.2, 0.5, 0.7]),
    "C3": np.array([0.5, 0.7, 0.7]),
    "C4": np.array([0.8, 0.8, 0.3]),
    "C5": np.array([0.5, 0.9, 0.7]),
}


def get_blended_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    phase_names=("B", "C1", "C2", "C3", "C4", "C5"),
    phase_colors=PHASE_COLORS,
    overall_opacity: float = 0.5,
) -> Image.Image:
    """Blend a grid of phase fractions over the corresponding grayscale image."""

    image_2d = image.squeeze()
    height, width = image_2d.shape
    n_rows, n_cols, _ = labels.shape
    cell_h = height / n_rows
    cell_w = width / n_cols

    overlay = np.zeros((height, width, 4), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            fractions = labels[i, j]
            color = np.zeros(3, dtype=np.float32)
            for k, phase in enumerate(phase_names):
                color += fractions[k] * phase_colors[phase]
            y0, y1 = round(i * cell_h), round((i + 1) * cell_h)
            x0, x1 = round(j * cell_w), round((j + 1) * cell_w)
            overlay[y0:y1, x0:x1, :3] = color
            overlay[y0:y1, x0:x1, 3] = overall_opacity

    overlay = np.clip(overlay, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8), "RGBA")
    original = Image.fromarray((np.clip(image_2d, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    return Image.alpha_composite(original, overlay_img)


def plot_ground_truth_vs_prediction(
    image: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot ground-truth and predicted fraction overlays side by side."""

    gt_blended = get_blended_overlay(image, gt_labels)
    pred_blended = get_blended_overlay(image, pred_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(gt_blended)
    axes[0].axis("off")
    axes[0].set_title("Ground truth")
    axes[1].imshow(pred_blended)
    axes[1].axis("off")
    axes[1].set_title("Prediction")

    patches = [mpatches.Patch(color=PHASE_COLORS[p], label=p) for p in PHASE_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_training_history(history, save_path: str | None = None, dpi: int = 300):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history["loss"], label="Training loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig
