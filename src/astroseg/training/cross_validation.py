"""Deterministic grouped cross-validation assignments."""

from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd

from astroseg.constants import TRAINABLE_ANNOTATION_STATUSES
from astroseg.io.manifest import load_manifest, validate_manifest


def assign_grouped_folds(
    manifest: pd.DataFrame,
    n_splits: int,
    group_column: str = "image_id",
    fold_column: str = "fold",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each image/well group wholly to one deterministic fold."""
    validate_manifest(manifest)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if group_column not in manifest.columns:
        raise ValueError(f"Cross-validation group column is missing: {group_column!r}")
    groups = manifest[group_column].astype(str).str.strip()
    if (groups == "").any():
        raise ValueError(f"Cross-validation group column {group_column!r} contains empty values")
    unique_groups = sorted(groups.unique())
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Grouped cross-validation requires at least {n_splits} groups; found {len(unique_groups)}"
        )
    rng = np.random.default_rng(seed)
    shuffled_groups = [unique_groups[index] for index in rng.permutation(len(unique_groups))]
    group_to_fold = {group: index % n_splits for index, group in enumerate(shuffled_groups)}
    result = manifest.copy()
    result[fold_column] = groups.map(group_to_fold).astype(int)
    return result


def split_grouped_fold(
    manifest: pd.DataFrame,
    validation_fold: int,
    group_column: str = "image_id",
    fold_column: str = "fold",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/validation manifest frames and verify group disjointness."""
    if fold_column not in manifest.columns:
        raise ValueError(f"Fold column is missing: {fold_column!r}")
    if group_column not in manifest.columns:
        raise ValueError(f"Cross-validation group column is missing: {group_column!r}")
    fold_values = pd.to_numeric(manifest[fold_column], errors="raise").astype(int)
    if validation_fold not in set(fold_values):
        raise ValueError(f"validation_fold {validation_fold} is not present in {fold_column!r}")
    validation = manifest.loc[fold_values == validation_fold].copy()
    training = manifest.loc[fold_values != validation_fold].copy()
    training_groups = set(training[group_column].astype(str))
    validation_groups = set(validation[group_column].astype(str))
    overlap = training_groups & validation_groups
    if overlap:
        raise ValueError(f"Cross-validation group leakage detected: {sorted(overlap)}")
    training["split"] = "train"
    validation["split"] = "val"
    return training, validation


def load_grouped_fold_manifests(
    manifest_path: str | Path,
    n_splits: int,
    validation_fold: int,
    group_column: str = "image_id",
    fold_column: str = "fold",
    seed: int = 42,
    annotation_statuses: Collection[str] = TRAINABLE_ANNOTATION_STATUSES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load eligible annotations and return one leakage-free grouped split."""
    if isinstance(annotation_statuses, str):
        raise TypeError("annotation_statuses must be a collection of states, not one string")
    manifest = load_manifest(manifest_path)
    normalized_statuses = {str(status).strip().lower() for status in annotation_statuses}
    candidates = manifest.loc[
        (manifest["split"] != "test") & manifest["annotation_status"].isin(normalized_statuses)
    ].copy()
    if candidates.empty:
        raise ValueError("No eligible annotated images are available for grouped cross-validation")
    if fold_column not in candidates.columns or (candidates[fold_column].astype(str).str.strip() == "").any():
        candidates = assign_grouped_folds(candidates, n_splits, group_column, fold_column, seed)
    return split_grouped_fold(candidates, validation_fold, group_column, fold_column)
