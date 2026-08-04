"""Ground-truth and prediction overlay writers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astroseg.preprocessing.normalize import percentile_normalize
from skimage.segmentation import find_boundaries


def save_segmentation_overlay(
    gfap_image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    title: str = "Segmentation overlay",
    color: tuple[float, float, float] = (1.0, 0.2, 0.1),
    alpha: float = 0.45,
) -> None:
    """Save a titled binary segmentation overlay on normalized GFAP intensity.

    Positive pixels blend the configured RGB color with the grayscale background,
    while negative pixels retain GFAP contrast. This function is for QC only.
    """
    if gfap_image.ndim != 2 or mask.shape != gfap_image.shape:
        raise ValueError("gfap_image and mask must be aligned 2D arrays")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = np.repeat(percentile_normalize(gfap_image)[..., None], 3, axis=2)
    overlay = base.copy()
    positive = mask.astype(bool)
    overlay[positive] = (1.0 - alpha) * base[positive] + alpha * np.asarray(color)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(np.clip(overlay, 0, 1))
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_instance_overlay(
    gfap_image: np.ndarray,
    instance_labels: np.ndarray,
    output_path: str | Path,
    title: str = "Individual astrocyte instances",
    alpha: float = 0.55,
) -> None:
    """Save uniquely colored cell IDs over the aligned GFAP channel.

    A deterministic golden-ratio hue maps arbitrary positive IDs to colors, and
    white boundary pixels make touching astrocytes easy to inspect.
    """
    if gfap_image.ndim != 2 or instance_labels.shape != gfap_image.shape:
        raise ValueError("GFAP and instance labels must be aligned 2D arrays")
    if np.any(instance_labels < 0) or not np.equal(instance_labels, np.floor(instance_labels)).all():
        raise ValueError("Instance labels must contain non-negative integers")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = np.repeat(percentile_normalize(gfap_image)[..., None], 3, axis=2)
    normalized_ids = (instance_labels.astype(np.float64) * 0.61803398875) % 1.0
    colors = plt.get_cmap("hsv")(normalized_ids)[..., :3]
    positive = instance_labels > 0
    overlay = base.copy()
    overlay[positive] = (1.0 - alpha) * base[positive] + alpha * colors[positive]
    overlay[find_boundaries(instance_labels, mode="outer")] = 1.0
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(np.clip(overlay, 0, 1))
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_compartment_overlay(
    gfap_image: np.ndarray,
    compartments: np.ndarray,
    output_path: str | Path,
    title: str = "Astrocyte compartments",
    alpha: float = 0.6,
) -> None:
    """Save nucleus, soma, and process classes in fixed biological colors.

    Blue denotes nucleus, orange soma, and green process; background retains the
    normalized GFAP image for spatial context.
    """
    if gfap_image.ndim != 2 or compartments.shape != gfap_image.shape:
        raise ValueError("GFAP and compartments must be aligned 2D arrays")
    if not set(np.unique(compartments)).issubset({0, 1, 2, 3}):
        raise ValueError("Compartment labels must use background=0, nucleus=1, soma=2, process=3")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = np.repeat(percentile_normalize(gfap_image)[..., None], 3, axis=2)
    overlay = base.copy()
    colors = {
        1: np.array([0.1, 0.45, 1.0]),
        2: np.array([1.0, 0.45, 0.05]),
        3: np.array([0.15, 0.9, 0.3]),
    }
    for class_id, color in colors.items():
        selected = compartments == class_id
        overlay[selected] = (1.0 - alpha) * base[selected] + alpha * color
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(np.clip(overlay, 0, 1))
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
