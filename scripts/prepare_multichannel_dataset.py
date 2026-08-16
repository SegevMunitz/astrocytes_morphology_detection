"""Build a validated three-fluorescence-channel cluster dataset snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.build_drive_manifest import build_drive_manifest
    from scripts.import_instance_annotations import import_instance_pairs
    from scripts.prepare_dataset import prepare_dataset
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_drive_manifest import build_drive_manifest
    from import_instance_annotations import import_instance_pairs
    from prepare_dataset import prepare_dataset


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a large artifact at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_masks(
    training_directory: Path,
    mask_directory: Path,
    training_ids: set[str],
) -> pd.DataFrame:
    """Validate authoritative masks and matching co-located Cellpose copies."""
    mask_paths = sorted(mask_directory.glob("*_seg.npy"))
    if not mask_paths:
        raise ValueError(f"No *_seg.npy masks found in {mask_directory}")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for mask_path in mask_paths:
        image_id = mask_path.name.removesuffix("_seg.npy")
        if image_id in seen:
            raise ValueError(f"Duplicate authoritative mask for {image_id!r}")
        if image_id not in training_ids:
            raise ValueError(f"Mask {mask_path.name!r} has no matching training image")
        seen.add(image_id)
        payload = np.load(mask_path, allow_pickle=True).item()
        labels = np.asarray(payload.get("masks"))
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Mask {mask_path.name!r} is not a 2D integer label image")
        instance_count = int(np.unique(labels[labels > 0]).size)
        if instance_count < 1:
            raise ValueError(f"Mask {mask_path.name!r} contains no instances")
        colocated = training_directory / mask_path.name
        if colocated.exists() and _sha256(colocated) != _sha256(mask_path):
            raise ValueError(
                f"Co-located and training_masks copies differ for {mask_path.name!r}; "
                "training_masks is authoritative, but the conflict must be resolved"
            )
        manual = np.asarray(payload.get("ismanual", []), dtype=bool)
        records.append(
            {
                "image_id": image_id,
                "mask_path": str(mask_path.resolve()),
                "height": int(labels.shape[0]),
                "width": int(labels.shape[1]),
                "instance_count": instance_count,
                "manual_instance_count": int(manual.sum()),
                "co_located_copy": colocated.is_file(),
                "sha256": _sha256(mask_path),
            }
        )
    return pd.DataFrame(records)


def prepare_multichannel_dataset(
    data_root: Path,
    destination: Path,
    annotator: str = "",
) -> pd.DataFrame:
    """Create manifests, nuclei, imported instances, QC, and audit metadata.

    The destination must be new so source images, reviewed masks, and earlier
    experimental snapshots cannot be overwritten accidentally.
    """
    data_root = data_root.resolve()
    training_directory = data_root / "training_images"
    mask_directory = data_root / "training_masks"
    test_directory = data_root / "test_images"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset snapshot: {destination}")
    destination.mkdir(parents=True)

    image_manifest = destination / "metadata" / "manifest_images.csv"
    prepared_manifest = destination / "metadata" / "manifest_prepared.csv"
    final_manifest = destination / "metadata" / "manifest_instances.csv"
    manifest = build_drive_manifest(training_directory, test_directory, image_manifest)
    training_ids = set(manifest.loc[manifest["split"] == "train", "image_id"])
    mask_report = _audit_masks(training_directory, mask_directory, training_ids)

    prepared = prepare_dataset(
        image_manifest,
        destination / "interim",
        prepared_manifest,
    )
    result = import_instance_pairs(
        prepared_manifest,
        None,
        destination / "annotations",
        final_manifest,
        "reviewed",
        "updated_cellpose_manual_review",
        annotator,
        cellpose_directory=mask_directory,
    )
    mask_report.to_csv(destination / "metadata" / "mask_audit.csv", index=False)
    summary = {
        "input_contract": ["GFAP/Cy5", "GFP_or_zero", "DAPI"],
        "training_images": int((result["split"] == "train").sum()),
        "labeled_training_images": int(
            ((result["split"] == "train") & (result["annotation_status"] == "reviewed")).sum()
        ),
        "unlabeled_training_images": int(
            ((result["split"] == "train") & (result["annotation_status"] == "none")).sum()
        ),
        "test_images": int((result["split"] == "test").sum()),
        "annotated_instances": int(mask_report["instance_count"].sum()),
        "manually_drawn_instances": int(mask_report["manual_instance_count"].sum()),
        "manifest": str(final_manifest.resolve()),
    }
    (destination / "metadata" / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    """Parse canonical cluster data root and versioned snapshot destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotator", default="")
    return parser.parse_args()


def main() -> None:
    """Prepare one immutable snapshot and print its final split counts."""
    args = parse_args()
    manifest = prepare_multichannel_dataset(args.data_root, args.output_dir, args.annotator)
    counts = manifest["split"].value_counts().to_dict()
    labeled = int((manifest["annotation_status"] == "reviewed").sum())
    print(f"Prepared {len(manifest)} images ({counts}); labeled={labeled}")


if __name__ == "__main__":
    main()
