"""Manifest schema loading and validation."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from astroseg.constants import ANNOTATION_STATUSES, MANIFEST_COLUMNS, VALID_SPLITS


class ManifestRow(BaseModel):
    """Serializable schema for a microscopy manifest row."""

    model_config = ConfigDict(extra="allow")

    image_id: str
    experiment_id: str = ""
    timepoint: str = ""
    treatment: str = ""
    magnification: str = ""
    path: str
    gfap_channel: str = ""
    dapi_channel: str = ""
    cellpose_mask_path: str = ""
    annotation_path: str = ""
    annotation_status: str = "none"
    annotation_source: str = ""
    annotator: str = ""
    review_status: str = ""
    split: str = ""


def validate_manifest(manifest: pd.DataFrame) -> None:
    """Validate required columns, identifiers, paths, splits, and annotation states."""
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    image_ids = manifest["image_id"].astype(str).str.strip()
    if (image_ids == "").any():
        raise ValueError("Manifest image_id values must not be empty")
    duplicated = image_ids[image_ids.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Manifest image_id values must be unique; duplicates: {duplicated}")
    paths = manifest["path"].astype(str).str.strip()
    if (paths == "").any():
        raise ValueError("Manifest path values must not be empty")
    splits = manifest["split"].astype(str).str.strip().str.lower()
    invalid = sorted(set(splits) - VALID_SPLITS)
    if invalid:
        raise ValueError(f"Invalid split values {invalid}; allowed values are train, val, test, or empty")
    statuses = manifest["annotation_status"].astype(str).str.strip().str.lower()
    invalid_statuses = sorted(set(statuses) - ANNOTATION_STATUSES)
    if invalid_statuses:
        raise ValueError(
            f"Invalid annotation_status values {invalid_statuses}; "
            f"allowed values are {sorted(ANNOTATION_STATUSES)}"
        )
    annotated = statuses != "none"
    missing_annotations = manifest.loc[annotated, "annotation_path"].astype(str).str.strip() == ""
    if missing_annotations.any():
        image_ids = manifest.loc[annotated].loc[missing_annotations, "image_id"].tolist()
        raise ValueError(f"Annotated rows require annotation_path values; missing for {image_ids}")


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a CSV manifest as strings and validate its required schema."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {source}")
    manifest = pd.read_csv(source, dtype=str, keep_default_na=False)
    validate_manifest(manifest)
    manifest["split"] = manifest["split"].str.strip().str.lower()
    manifest["annotation_status"] = manifest["annotation_status"].str.strip().str.lower()
    return manifest
