"""Create binary nucleus masks, proximity maps, and preprocessing QC images."""

import argparse
from pathlib import Path

import numpy as np
import tifffile

from astroseg.io import get_channel, load_manifest, load_ome_tiff
from astroseg.preprocessing import (
    create_nucleus_proximity_map,
    labels_to_binary_mask,
    validate_nucleus_labels,
)
from astroseg.visualization import (
    save_nucleus_label_preview,
    save_nucleus_mask_preview,
    save_proximity_map_preview,
    save_qc_montage,
)


def _resolve_path(value: str, manifest_path: Path, description: str) -> Path:
    """Resolve an image or mask path while retaining a descriptive error label.

    Direct project-relative and manifest-relative candidates are supported.
    Missing paths raise before any derived nucleus artifacts are written.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} path does not exist: {value}")


def _load_labels(path: Path) -> np.ndarray:
    """Load a Cellpose label array from NumPy or TIFF storage.

    The array is returned unchanged for shared structural validation; unsupported
    formats naturally fail instead of being interpreted through filename guessing.
    """
    labels = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else tifffile.imread(path)
    return np.asarray(labels)


def generate_nucleus_inputs(
    manifest_path: Path,
    output_directory: Path,
    max_distance: float = 64.0,
) -> int:
    """Generate binary nucleus and proximity inputs for every manifest image.

    Cellpose labels are shape-validated against OME dimensions before NumPy arrays,
    individual previews, and an optional GFAP-aligned QC montage are written.
    """
    manifest = load_manifest(manifest_path)
    mask_directory = output_directory / "nucleus_masks"
    distance_directory = output_directory / "nucleus_distance_maps"
    qc_directory = output_directory / "qc"
    count = 0
    for _, row in manifest.iterrows():
        if not row["cellpose_mask_path"].strip():
            raise ValueError(f"cellpose_mask_path is empty for {row['image_id']!r}")
        microscopy = load_ome_tiff(_resolve_path(row["path"], manifest_path, "Image"))
        labels = _load_labels(_resolve_path(row["cellpose_mask_path"], manifest_path, "Cellpose mask"))
        validate_nucleus_labels(labels, microscopy.image.shape[-2:])
        binary = labels_to_binary_mask(labels)
        proximity = create_nucleus_proximity_map(binary, max_distance)
        mask_directory.mkdir(parents=True, exist_ok=True)
        distance_directory.mkdir(parents=True, exist_ok=True)
        np.save(mask_directory / f"{row['image_id']}.npy", binary, allow_pickle=False)
        np.save(distance_directory / f"{row['image_id']}.npy", proximity, allow_pickle=False)
        save_nucleus_label_preview(labels, qc_directory / f"{row['image_id']}_labels.png")
        save_nucleus_mask_preview(binary, qc_directory / f"{row['image_id']}_mask.png")
        save_proximity_map_preview(proximity, qc_directory / f"{row['image_id']}_proximity.png")
        if row["gfap_channel"].strip():
            gfap = get_channel(microscopy, row["gfap_channel"])
            save_qc_montage(
                gfap,
                labels,
                binary,
                proximity,
                qc_directory / f"{row['image_id']}_montage.png",
            )
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    """Parse nucleus-input generation paths and distance cutoff.

    Manifest and output directory are required, while maximum distance defaults
    to the same 64-pixel value used by the baseline data configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=64.0)
    return parser.parse_args()


def main() -> None:
    """Generate all nucleus-derived artifacts and report the image count.

    Errors identify missing masks or shape mismatches, so the printed completion
    summary is emitted only after every selected manifest row succeeds.
    """
    args = parse_args()
    count = generate_nucleus_inputs(args.manifest, args.output_dir, args.max_distance)
    print(f"Generated nucleus inputs for {count} images in {args.output_dir}")


if __name__ == "__main__":
    main()
