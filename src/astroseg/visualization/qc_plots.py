"""Quality-control preview and montage writers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_preview(image: np.ndarray, output_path: str | Path, cmap: str, title: str) -> None:
    if image.ndim != 2:
        raise ValueError("preview image must be 2D")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_gfap_preview(image: np.ndarray, output_path: str | Path) -> None:
    """Save a grayscale GFAP preview."""
    _save_preview(image, output_path, "gray", "GFAP")


def save_nucleus_label_preview(labels: np.ndarray, output_path: str | Path) -> None:
    """Save a categorical Cellpose nucleus-label preview."""
    _save_preview(labels, output_path, "nipy_spectral", "Nucleus labels")


def save_nucleus_mask_preview(mask: np.ndarray, output_path: str | Path) -> None:
    """Save a binary nucleus-mask preview."""
    _save_preview(mask, output_path, "gray", "Nucleus mask")


def save_proximity_map_preview(proximity: np.ndarray, output_path: str | Path) -> None:
    """Save a nucleus-proximity heat-map preview."""
    _save_preview(proximity, output_path, "magma", "Nucleus proximity")


def save_qc_montage(
    gfap: np.ndarray,
    nucleus_labels: np.ndarray,
    nucleus_mask: np.ndarray,
    proximity_map: np.ndarray,
    output_path: str | Path,
    ground_truth: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
) -> None:
    """Save a compact montage of preprocessing and optional segmentation outputs."""
    arrays = [gfap, nucleus_labels, nucleus_mask, proximity_map]
    if any(array.ndim != 2 or array.shape != gfap.shape for array in arrays):
        raise ValueError("All QC inputs must be aligned 2D arrays")
    panels: list[tuple[str, np.ndarray, str]] = [
        ("GFAP", gfap, "gray"),
        ("Nucleus labels", nucleus_labels, "nipy_spectral"),
        ("Nucleus mask", nucleus_mask, "gray"),
        ("Proximity", proximity_map, "magma"),
    ]
    for title, optional in (("Ground truth", ground_truth), ("Prediction", prediction)):
        if optional is not None:
            if optional.shape != gfap.shape:
                raise ValueError(f"{title} shape does not match GFAP shape")
            panels.append((title, optional, "viridis"))
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for axis, (title, array, cmap) in zip(axes.flat, panels, strict=False):
        axis.imshow(array, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
