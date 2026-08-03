"""Create a lightweight CSV index of patch coordinates without copying image arrays."""

import argparse
from pathlib import Path

import pandas as pd

from astroseg.io import load_manifest, load_ome_tiff
from astroseg.preprocessing import generate_patch_coordinates


def _resolve_path(value: str, manifest_path: Path) -> Path:
    """Resolve a manifest image reference before reading OME dimensions.

    Project-relative and manifest-relative candidates are supported; failure keeps
    the original manifest value visible for straightforward correction.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Manifest image path does not exist: {value}")


def create_patch_index(
    manifest_path: Path,
    output_path: Path,
    patch_size: int = 512,
    overlap: int = 64,
) -> pd.DataFrame:
    """Create a coordinate-only patch index for every manifest image.

    Image dimensions are read from OME-TIFF files, but pixel arrays are never
    duplicated. Each output row retains split and annotation-status metadata.
    """
    manifest = load_manifest(manifest_path)
    records = []
    for _, row in manifest.iterrows():
        image = load_ome_tiff(_resolve_path(row["path"], manifest_path))
        for patch_index, coordinates in enumerate(
            generate_patch_coordinates(image.image.shape[-2:], patch_size, overlap)
        ):
            records.append(
                {
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "annotation_status": row["annotation_status"],
                    "patch_index": patch_index,
                    "y": coordinates.y,
                    "x": coordinates.x,
                    "height": coordinates.height,
                    "width": coordinates.width,
                }
            )
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    """Parse manifest, output, patch-size, and overlap command arguments.

    Filesystem values become ``Path`` objects and numeric patch parameters retain
    explicit defaults. Detailed validation occurs in the patch library.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    """Build the patch index and print its destination and row count.

    The command delegates all coordinate generation to the tested preprocessing
    implementation, ensuring CLI and dataset grids remain consistent.
    """
    args = parse_args()
    frame = create_patch_index(args.manifest, args.output, args.patch_size, args.overlap)
    print(f"Wrote {len(frame)} patch rows to {args.output}")


if __name__ == "__main__":
    main()
