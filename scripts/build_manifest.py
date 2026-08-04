"""Discover microscopy TIFF/BMP images and create a manifest template."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from astroseg.constants import MANIFEST_COLUMNS, MICROSCOPY_IMAGE_SUFFIXES


def _display_path(path: Path) -> str:
    """Format a discovered image path for portable manifest storage.

    Files under the working tree become relative POSIX-style paths, while external
    discovery roots retain explicit paths because they cannot be made project local.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(raw_directory: Path, output_path: Path) -> pd.DataFrame:
    """Discover TIFF/BMP images and write a conservative manifest template.

    Stable IDs derive from stems, with path hashes resolving duplicate stems.
    Experimental metadata and channel identities remain empty for manual entry.
    """
    if not raw_directory.is_dir():
        raise NotADirectoryError(f"Raw-data directory does not exist: {raw_directory}")
    files = sorted(
        path
        for path in raw_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in MICROSCOPY_IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No .bmp, .tif, or .tiff files were found under {raw_directory}")
    stem_counts: dict[str, int] = {}
    for path in files:
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1
    rows = []
    for path in files:
        image_id = path.stem
        if stem_counts[path.stem] > 1:
            relative = path.relative_to(raw_directory).as_posix()
            suffix = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
            image_id = f"{path.stem}_{suffix}"
        row = {column: "" for column in MANIFEST_COLUMNS}
        row.update({"image_id": image_id, "path": _display_path(path), "annotation_status": "none"})
        rows.append(row)
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Define and parse the manifest-builder command-line interface.

    Both the raw discovery directory and destination CSV are required and returned
    as ``Path`` objects. No filesystem work occurs during argument parsing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run microscopy-image discovery and report generated manifest rows.

    Validation errors propagate with context so an empty or incorrect raw-data
    directory cannot produce a misleading empty catalog.
    """
    args = parse_args()
    manifest = build_manifest(args.raw_dir, args.output)
    print(f"Wrote {len(manifest)} image rows to {args.output}")


if __name__ == "__main__":
    main()
