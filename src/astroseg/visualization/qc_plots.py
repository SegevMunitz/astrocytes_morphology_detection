"""Quality-control preview and montage writers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_preview(image: np.ndarray, output_path: str | Path, cmap: str, title: str) -> None:
    """Render one validated two-dimensional array with a titled color map.

    The non-interactive backend writes a compact PNG and closes its figure promptly,
    allowing batch preprocessing without accumulating GUI or figure resources.
    """
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
    """Save a consistently styled grayscale preview of the GFAP channel.

    The function validates two-dimensional input, hides axes, and writes a titled
    PNG suitable for quick channel-selection quality control.
    """
    save_grayscale_preview(image, output_path, "GFAP")


def save_grayscale_preview(
    image: np.ndarray, output_path: str | Path, title: str
) -> None:
    """Save any two-dimensional microscopy channel as a grayscale preview.

    The explicit title distinguishes channels such as DAPI from GFAP while sharing
    the same visualization behavior. No intensity values are changed on disk.
    """
    if not title.strip():
        raise ValueError("Preview title must not be empty")
    _save_preview(image, output_path, "gray", title)


def save_nucleus_label_preview(labels: np.ndarray, output_path: str | Path) -> None:
    """Save a categorical preview of Cellpose nucleus instance identifiers.

    A spectral visualization separates adjacent integer instances visually; the
    colors are display-only and have no role in preprocessing or analysis.
    """
    _save_preview(labels, output_path, "nipy_spectral", "Nucleus labels")


def save_nucleus_mask_preview(mask: np.ndarray, output_path: str | Path) -> None:
    """Save a grayscale preview of the derived binary nucleus mask.

    The output helps verify that all positive Cellpose instances became foreground
    and that the mask remains aligned with the microscopy field.
    """
    _save_preview(mask, output_path, "gray", "Nucleus mask")


def save_proximity_map_preview(proximity: np.ndarray, output_path: str | Path) -> None:
    """Save a heat-map preview of the nucleus-proximity model input.

    The visualization should peak on nucleus pixels and decay with distance,
    making unexpected empty or misaligned distance maps easy to detect.
    """
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
    """Save a compact QC montage of aligned preprocessing and segmentation planes.

    GFAP, nucleus labels, binary mask, and proximity are always shown; ground truth
    and prediction panels are appended when provided. All arrays must share shape.
    """
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
