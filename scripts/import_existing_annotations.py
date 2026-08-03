"""Import corrected Cellpose masks non-destructively and update annotation metadata."""

import argparse
from pathlib import Path

import pandas as pd

from astroseg.annotations import import_annotation_pair
from astroseg.constants import HUMAN_ANNOTATION_STATUSES
from astroseg.io import load_manifest, validate_manifest


def _resolve_path(value: str, base_directory: Path, description: str) -> Path:
    path = Path(value)
    for candidate in (path, base_directory / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} does not exist: {value}")


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def import_pairs(
    manifest_path: Path,
    pairs_path: Path,
    output_directory: Path,
    output_manifest: Path,
    default_status: str,
    default_source: str,
    default_annotator: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Import pair-table masks and write an updated manifest."""
    manifest = load_manifest(manifest_path)
    pairs = pd.read_csv(pairs_path, dtype=str, keep_default_na=False)
    required = {"image_id", "mask_path"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Annotation-pair CSV is missing required columns: {sorted(missing)}")
    if pairs["image_id"].duplicated().any():
        raise ValueError("Annotation-pair CSV contains duplicate image_id values")
    for _, pair in pairs.iterrows():
        matches = manifest.index[manifest["image_id"] == pair["image_id"]].tolist()
        if len(matches) != 1:
            raise ValueError(f"Pair image_id {pair['image_id']!r} does not identify one manifest row")
        index = matches[0]
        row = manifest.loc[index]
        if row["annotation_status"] != "none" and not overwrite:
            raise ValueError(
                f"Image {pair['image_id']!r} already has annotation status {row['annotation_status']!r}; "
                "use --overwrite to import a later lifecycle state"
            )
        status = pair.get("annotation_status", "").strip().lower() or default_status
        source = pair.get("annotation_source", "").strip() or default_source
        annotator = pair.get("annotator", "").strip() or default_annotator
        review_status = pair.get("review_status", "").strip()
        result = import_annotation_pair(
            str(row["image_id"]),
            _resolve_path(str(row["path"]), manifest_path.parent, "Microscopy image"),
            _resolve_path(str(pair["mask_path"]), pairs_path.parent, "Annotation mask"),
            str(row["gfap_channel"]),
            output_directory,
            status,
            source,
            annotator,
            review_status,
            overwrite,
        )
        manifest.loc[index, "annotation_path"] = _portable_path(result.binary_mask_path)
        manifest.loc[index, "annotation_status"] = result.annotation_status
        manifest.loc[index, "annotation_source"] = result.annotation_source
        manifest.loc[index, "annotator"] = result.annotator
        manifest.loc[index, "review_status"] = result.review_status
    validate_manifest(manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--status", choices=sorted(HUMAN_ANNOTATION_STATUSES), default="seed")
    parser.add_argument("--source", default="manual_cellpose_correction")
    parser.add_argument("--annotator", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Import annotations and report the updated manifest."""
    args = parse_args()
    manifest = import_pairs(
        args.manifest,
        args.pairs_csv,
        args.output_dir,
        args.output_manifest,
        args.status,
        args.source,
        args.annotator,
        args.overwrite,
    )
    print(f"Wrote {len(manifest)} manifest rows to {args.output_manifest}")


if __name__ == "__main__":
    main()
