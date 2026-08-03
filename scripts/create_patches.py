"""Create a lightweight CSV index of patch coordinates without copying image arrays."""

import argparse
from pathlib import Path

import pandas as pd

from astroseg.io import load_manifest, load_ome_tiff
from astroseg.preprocessing import generate_patch_coordinates


def _resolve_path(value: str, manifest_path: Path) -> Path:
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
    """Write one coordinate row per image patch."""
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    """Create the patch-index CSV."""
    args = parse_args()
    frame = create_patch_index(args.manifest, args.output, args.patch_size, args.overlap)
    print(f"Wrote {len(frame)} patch rows to {args.output}")


if __name__ == "__main__":
    main()
