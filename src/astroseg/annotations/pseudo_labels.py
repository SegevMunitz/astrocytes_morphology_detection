"""Storage helpers for automatic labels kept outside human annotation data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from astroseg.visualization.overlays import save_segmentation_overlay


@dataclass(frozen=True)
class PseudoLabelArtifacts:
    """Automatic probability, mask, and QC paths for one image."""

    probability_path: Path
    mask_path: Path
    overlay_path: Path


def save_pseudo_label_artifacts(
    image_id: str,
    probabilities: np.ndarray,
    gfap_image: np.ndarray,
    output_directory: str | Path,
    overwrite: bool = False,
) -> PseudoLabelArtifacts:
    """Save automatic probabilities and masks separately from human annotations."""
    if not image_id.strip():
        raise ValueError("image_id must not be empty")
    if probabilities.ndim != 3 or probabilities.shape[0] != 2:
        raise ValueError("Binary pseudo-label probabilities must have shape [2, H, W]")
    if probabilities.shape[-2:] != gfap_image.shape or gfap_image.ndim != 2:
        raise ValueError("Probability and GFAP image dimensions must match")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-4):
        raise ValueError("Class probabilities must sum to one at every pixel")

    destination = Path(output_directory)
    probability_path = destination / "probabilities" / f"{image_id}.npy"
    mask_path = destination / "masks" / f"{image_id}.tiff"
    overlay_path = destination / "overlays" / f"{image_id}.png"
    for path in (probability_path, mask_path, overlay_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite pseudo-label artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    mask = probabilities.argmax(axis=0).astype(np.uint8)
    np.save(probability_path, probabilities.astype(np.float32, copy=False), allow_pickle=False)
    tifffile.imwrite(mask_path, mask)
    save_segmentation_overlay(gfap_image, mask > 0, overlay_path, title="Pseudo-label overlay")
    return PseudoLabelArtifacts(probability_path, mask_path, overlay_path)

