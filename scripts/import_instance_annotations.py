"""Import human-corrected complete astrocyte instances and update a new manifest."""

import argparse
from pathlib import Path

import pandas as pd

from astroseg.annotations import import_astrocyte_instance_pair
from astroseg.constants import HUMAN_ANNOTATION_STATUSES
from astroseg.io import load_manifest, validate_manifest


def _resolve_path(value: str, base_directory: Path, description: str) -> Path:
    """Resolve explicit project- or table-relative paths with readable errors.

    No filename inference is used because pairing the wrong cell mask is a silent
    and scientifically serious ownership-label error.
    """
    path = Path(value)
    for candidate in (path, base_directory / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} does not exist: {value}")


def _portable_path(path: Path | None) -> str:
    """Convert imported artifacts to portable project-relative paths.

    A missing optional compartment mask remains an empty manifest value.
    """
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def import_instance_pairs(
    manifest_path: Path,
    pairs_path: Path,
    output_directory: Path,
    output_manifest: Path,
    default_status: str,
    default_source: str,
    default_annotator: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Import all image/instance pairs and record authoritative training paths.

    The output manifest retains human annotation lifecycle fields. Automatic
    predictions are never copied into these columns by this workflow.
    """
    manifest = load_manifest(manifest_path)
    pairs = pd.read_csv(pairs_path, dtype=str, keep_default_na=False)
    required = {"image_id", "instance_mask_path"}
    missing = required - set(pairs)
    if missing:
        raise ValueError(f"Instance-pair CSV is missing columns: {sorted(missing)}")
    if pairs["image_id"].duplicated().any():
        raise ValueError("Instance-pair CSV contains duplicate image_id values")
    for _, pair in pairs.iterrows():
        indices = manifest.index[manifest["image_id"] == pair["image_id"]].tolist()
        if len(indices) != 1:
            raise ValueError(f"Pair image_id {pair['image_id']!r} does not identify one row")
        index = indices[0]
        row = manifest.loc[index]
        if str(row["annotation_status"]) != "none" and not overwrite:
            raise ValueError(
                f"Image {pair['image_id']!r} already has annotation status "
                f"{row['annotation_status']!r}; use --overwrite for a corrected version"
            )
        status = pair.get("annotation_status", "").strip().lower() or default_status
        source = pair.get("annotation_source", "").strip() or default_source
        annotator = pair.get("annotator", "").strip() or default_annotator
        compartment_value = pair.get("compartment_mask_path", "").strip()
        result = import_astrocyte_instance_pair(
            str(row["image_id"]),
            _resolve_path(str(row["path"]), manifest_path.parent, "Microscopy image"),
            _resolve_path(
                str(row["cellpose_mask_path"]), manifest_path.parent, "Nucleus mask"
            ),
            _resolve_path(
                str(pair["instance_mask_path"]), pairs_path.parent, "Instance annotation"
            ),
            str(row["gfap_channel"]),
            output_directory,
            (
                _resolve_path(compartment_value, pairs_path.parent, "Compartment annotation")
                if compartment_value
                else None
            ),
            status,
            source,
            annotator,
            pair.get("review_status", "").strip(),
            overwrite=overwrite,
        )
        manifest.loc[index, "annotation_path"] = _portable_path(result.base.binary_mask_path)
        manifest.loc[index, "instance_annotation_path"] = _portable_path(
            result.instance_mask_path
        )
        manifest.loc[index, "compartment_annotation_path"] = _portable_path(
            result.compartment_mask_path
        )
        manifest.loc[index, "annotation_status"] = result.base.annotation_status
        manifest.loc[index, "annotation_source"] = result.base.annotation_source
        manifest.loc[index, "annotator"] = result.base.annotator
        manifest.loc[index, "review_status"] = result.base.review_status
    validate_manifest(manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the pair table, lifecycle provenance, and non-destructive destinations.

    Per-row provenance columns override global defaults when supplied.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--status", choices=sorted(HUMAN_ANNOTATION_STATUSES), default="seed")
    parser.add_argument("--source", default="manual_complete_cell_correction")
    parser.add_argument("--annotator", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the complete-cell import and report the new manifest location.

    Existing manifests and source exports are never modified in place.
    """
    args = parse_args()
    manifest = import_instance_pairs(
        args.manifest,
        args.pairs_csv,
        args.output_dir,
        args.output_manifest,
        args.status,
        args.source,
        args.annotator,
        args.overwrite,
    )
    print(f"Wrote {len(manifest)} rows with complete-cell annotation metadata")


if __name__ == "__main__":
    main()
