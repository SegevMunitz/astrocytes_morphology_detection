"""Build one manifest from locally synchronized Drive train and test folders."""

import argparse
from pathlib import Path

import pandas as pd

from astroseg.constants import MANIFEST_COLUMNS, MICROSCOPY_IMAGE_SUFFIXES


def _portable_path(path: Path) -> str:
    """Represent a synchronized image relative to the working tree when possible.

    External workspaces retain absolute paths, while the default ignored runtime
    directory remains portable across repository clones.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_drive_manifest(
    training_directory: Path,
    test_directory: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Catalog synchronized Drive images with explicit train/test assignments.

    Exact image stems become IDs so ``<image_id>_seg.npy`` masks pair safely.
    Duplicate stems across either folder fail instead of receiving opaque hashes.
    """
    sources = ((training_directory, "train"), (test_directory, "test"))
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for directory, split in sources:
        if not directory.is_dir():
            raise NotADirectoryError(f"Synchronized Drive directory is missing: {directory}")
        files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in MICROSCOPY_IMAGE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No supported microscopy images found in {directory}")
        for path in files:
            image_id = path.stem
            if image_id in seen:
                raise ValueError(f"Duplicate image_id across Drive folders: {image_id!r}")
            seen.add(image_id)
            row = {column: "" for column in MANIFEST_COLUMNS}
            row.update(
                {
                    "image_id": image_id,
                    "path": _portable_path(path),
                    "annotation_status": "none",
                    "split": split,
                }
            )
            records.append(row)
    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse synchronized directories and ignored runtime manifest destination.

    Defaults match ``sync_google_drive.ps1 -Action Download`` exactly.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(".astroseg_runtime/dataset")
    parser.add_argument("--training-images", type=Path, default=root / "training_images")
    parser.add_argument("--test-images", type=Path, default=root / "test_images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".astroseg_runtime/outputs/metadata/manifest.csv"),
    )
    return parser.parse_args()


def main() -> None:
    """Write the remote-backed manifest and print its split counts.

    No Drive file or source image is modified by this operation.
    """
    args = parse_args()
    manifest = build_drive_manifest(args.training_images, args.test_images, args.output)
    counts = manifest["split"].value_counts().to_dict()
    print(f"Wrote {len(manifest)} rows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
