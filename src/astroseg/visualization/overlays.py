"""Ground-truth and prediction overlay writers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astroseg.preprocessing.normalize import percentile_normalize


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
