"""Uncertainty-based selection of unlabeled image patches."""

from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd

from astroseg.constants import ANNOTATION_STATUSES
from astroseg.io.manifest import validate_manifest
from astroseg.preprocessing.patches import generate_patch_coordinates


def select_uncertain_patches(
    manifest: pd.DataFrame,
    probability_directory: str | Path,
    patch_size: int,
    overlap: int,
    top_k: int,
    annotation_statuses: Collection[str] = ("none", "pseudo"),
    max_patches_per_image: int | None = None,
) -> pd.DataFrame:
    """Rank unreviewed patches by mean normalized predictive entropy."""
    validate_manifest(manifest)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_patches_per_image is not None and max_patches_per_image <= 0:
        raise ValueError("max_patches_per_image must be positive when provided")
    statuses = {str(status).strip().lower() for status in annotation_statuses}
    invalid = statuses - ANNOTATION_STATUSES
    if invalid:
        raise ValueError(f"Invalid selection annotation statuses: {sorted(invalid)}")
    candidates = manifest.loc[manifest["annotation_status"].isin(statuses)]
    if candidates.empty:
        raise ValueError("No manifest rows match the requested unlabeled annotation states")

    probability_root = Path(probability_directory)
    records: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        probability_path = probability_root / f"{row['image_id']}.npy"
        if not probability_path.is_file():
            raise FileNotFoundError(f"Probability map does not exist: {probability_path}")
        probabilities = np.load(probability_path, allow_pickle=False)
        if probabilities.ndim != 3 or probabilities.shape[0] < 2:
            raise ValueError(f"Probability map must have shape [C>=2, H, W]: {probability_path}")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError(f"Probability map contains invalid values: {probability_path}")
        sums = probabilities.sum(axis=0)
        if not np.allclose(sums, 1.0, atol=1e-4):
            raise ValueError(f"Probabilities do not sum to one: {probability_path}")
        clipped = np.clip(probabilities, 1e-7, 1.0)
        entropy = -(clipped * np.log(clipped)).sum(axis=0) / np.log(probabilities.shape[0])
        coordinates = generate_patch_coordinates(entropy.shape, patch_size, overlap)
        image_records = []
        for coordinate in coordinates:
            patch_entropy = entropy[
                coordinate.y : coordinate.y + coordinate.height,
                coordinate.x : coordinate.x + coordinate.width,
            ]
            image_records.append(
                {
                    "image_id": row["image_id"],
                    "annotation_status": row["annotation_status"],
                    "y": coordinate.y,
                    "x": coordinate.x,
                    "height": coordinate.height,
                    "width": coordinate.width,
                    "uncertainty": float(patch_entropy.mean()),
                    "probability_path": probability_path.as_posix(),
                }
            )
        image_records.sort(key=lambda item: (-float(item["uncertainty"]), int(item["y"]), int(item["x"])))
        if max_patches_per_image is not None:
            image_records = image_records[:max_patches_per_image]
        records.extend(image_records)
    records.sort(
        key=lambda item: (
            -float(item["uncertainty"]),
            str(item["image_id"]),
            int(item["y"]),
            int(item["x"]),
        )
    )
    return pd.DataFrame(records[:top_k])

