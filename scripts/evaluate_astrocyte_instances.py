"""Evaluate predicted complete astrocytes against human instance annotations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from astroseg.io import load_manifest
from astroseg.metrics import instance_segmentation_metrics, process_ownership_accuracy


def _resolve_path(value: str, manifest_path: Path, description: str) -> Path:
    """Resolve an explicit path relative to project or prediction manifest.

    Evaluation never guesses matching filenames because a mismatched mask can
    produce plausible but invalid object-level metrics.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} does not exist: {value}")


def _load_array(path: Path) -> np.ndarray:
    """Load one NumPy/TIFF two-dimensional label plane.

    Metric functions perform the remaining alignment and integer-label checks.
    """
    return np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else tifffile.imread(path)


def evaluate_manifest(
    manifest_path: Path,
    output_directory: Path,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate every row containing both human truth and automatic instances.

    Per-image CSV, matched-object JSON, and aggregate means are stored separately
    from training checkpoints and prediction artifacts.
    """
    manifest = load_manifest(manifest_path)
    required_columns = {"predicted_instance_path", "instance_annotation_path"}
    missing = required_columns - set(manifest)
    if missing:
        raise ValueError(f"Prediction manifest is missing columns: {sorted(missing)}")
    eligible = manifest[
        (manifest["predicted_instance_path"].astype(str).str.strip() != "")
        & (manifest["instance_annotation_path"].astype(str).str.strip() != "")
    ]
    if eligible.empty:
        raise ValueError("No rows contain both predicted and human instance paths")
    records: list[dict[str, object]] = []
    match_records: dict[str, object] = {}
    for _, row in eligible.iterrows():
        prediction = _load_array(
            _resolve_path(str(row["predicted_instance_path"]), manifest_path, "Prediction")
        )
        target = _load_array(
            _resolve_path(str(row["instance_annotation_path"]), manifest_path, "Target")
        )
        metrics = instance_segmentation_metrics(prediction, target, iou_threshold)
        matches = metrics.pop("matches")
        record: dict[str, object] = {"image_id": str(row["image_id"]), **metrics}
        compartment_value = str(row.get("compartment_annotation_path", "")).strip()
        if compartment_value:
            compartments = _load_array(
                _resolve_path(compartment_value, manifest_path, "Compartment target")
            )
            record["process_ownership_accuracy"] = process_ownership_accuracy(
                prediction, target, compartments
            )
        else:
            record["process_ownership_accuracy"] = float("nan")
        records.append(record)
        match_records[str(row["image_id"])] = matches
    result = pd.DataFrame(records)
    output_directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_directory / "instance_metrics_by_image.csv", index=False)
    numeric = result.select_dtypes(include="number")
    numeric.mean().to_frame("mean").T.to_csv(
        output_directory / "instance_metrics_summary.csv", index=False
    )
    with (output_directory / "instance_matches.json").open("w", encoding="utf-8") as handle:
        json.dump(match_records, handle, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    """Parse prediction manifest, output directory, and object IoU threshold.

    The default output location remains isolated from pixelwise semantic metrics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics/instances"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Run object evaluation and report the number of human-labeled images.

    Scientific interpretation remains with grouped held-out folds, not bootstrap data.
    """
    args = parse_args()
    result = evaluate_manifest(args.manifest, args.output_dir, args.iou_threshold)
    print(f"Evaluated complete-cell instances for {len(result)} image(s)")


if __name__ == "__main__":
    main()
