"""Extract explicitly named GFAP and DAPI channels from manifest images."""

import argparse
from pathlib import Path

import numpy as np

from astroseg.io import get_channel, load_manifest, load_ome_tiff
from astroseg.visualization import save_gfap_preview, save_grayscale_preview


def _resolve_path(value: str, manifest_path: Path) -> Path:
    """Resolve a manifest image path against supported relative locations.

    Project-relative paths are tried directly, followed by paths relative to the
    manifest directory. Missing files fail with the original manifest value.
    """
    path = Path(value)
    if path.is_file():
        return path
    candidate = manifest_path.parent / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Manifest image path does not exist: {value}")


def extract_channels(manifest_path: Path, output_directory: Path) -> int:
    """Extract explicitly named GFAP and DAPI channels for all manifest rows.

    Original-dtype NumPy arrays and labeled grayscale previews are written per
    image. Missing or ambiguous channel names stop processing clearly; standard
    RGB TIFFs expose the explicit names Red, Green, and Blue.
    """
    manifest = load_manifest(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    preview_directory = output_directory / "previews"
    count = 0
    for _, row in manifest.iterrows():
        if not row["gfap_channel"].strip() or not row["dapi_channel"].strip():
            raise ValueError(f"Explicit GFAP and DAPI channel names are required for {row['image_id']!r}")
        microscopy = load_ome_tiff(_resolve_path(row["path"], manifest_path))
        gfap = get_channel(microscopy, row["gfap_channel"])
        dapi = get_channel(microscopy, row["dapi_channel"])
        np.save(output_directory / f"{row['image_id']}_gfap.npy", gfap, allow_pickle=False)
        np.save(output_directory / f"{row['image_id']}_dapi.npy", dapi, allow_pickle=False)
        save_gfap_preview(gfap, preview_directory / f"{row['image_id']}_gfap.png")
        save_grayscale_preview(
            dapi, preview_directory / f"{row['image_id']}_dapi.png", "DAPI"
        )
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    """Parse the channel-extraction manifest and destination directory.

    Both paths are mandatory because the script never assumes data locations.
    The returned namespace performs no image I/O itself.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Extract manifest channels and print the number of processed images.

    Library exceptions remain visible to identify the precise image or missing
    channel rather than producing partial success messages.
    """
    args = parse_args()
    count = extract_channels(args.manifest, args.output_dir)
    print(f"Extracted channels for {count} images into {args.output_dir}")


if __name__ == "__main__":
    main()
