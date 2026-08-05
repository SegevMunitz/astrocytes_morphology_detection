"""Validate Cellpose GUI annotations and stage immutable image-mask pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


IMAGE_SUFFIXES = (".tif", ".tiff", ".bmp")


def _find_image(directory: Path, image_id: str) -> Path:
    """Return the unique supported image matching a ``*_seg.npy`` annotation."""
    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.stem == image_id
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {image_id!r}, found {len(matches)}")
    return matches[0]


def _spatial_shape(image: np.ndarray) -> tuple[int, int]:
    """Infer Y/X dimensions for grayscale, channel-first, or channel-last images."""
    if image.ndim == 2:
        return tuple(image.shape)
    if image.ndim != 3:
        raise ValueError(f"Cellpose training expects 2D images, received {image.shape}")
    if image.shape[-1] <= 4:
        return tuple(image.shape[:2])
    if image.shape[0] <= 4:
        return tuple(image.shape[-2:])
    raise ValueError(f"Cannot identify the channel axis for image shape {image.shape}")


def stage_pairs(
    input_directory: Path,
    staging_directory: Path,
    expected_channels: tuple[int, int],
    image_ids: set[str] | None = None,
) -> tuple[int, int]:
    """Validate every GUI mask and symlink each exact source pair into a run folder."""
    input_directory = input_directory.resolve()
    staging_directory.mkdir(parents=True, exist_ok=False)
    mask_paths = sorted(input_directory.glob("*_seg.npy"))
    if image_ids is not None:
        available = {path.name.removesuffix("_seg.npy") for path in mask_paths}
        missing = image_ids - available
        if missing:
            raise ValueError(f"Image IDs have no matching GUI masks: {sorted(missing)}")
        mask_paths = [
            path for path in mask_paths if path.name.removesuffix("_seg.npy") in image_ids
        ]
    if not mask_paths:
        raise ValueError(f"No *_seg.npy files found in {input_directory}")

    total_instances = 0
    for mask_path in mask_paths:
        image_id = mask_path.name.removesuffix("_seg.npy")
        image_path = _find_image(input_directory, image_id)
        payload = np.load(mask_path, allow_pickle=True).item()
        if "masks" not in payload:
            raise ValueError(f"{mask_path.name} has no 'masks' array")
        masks = np.asarray(payload["masks"])
        image_shape = _spatial_shape(tifffile.imread(image_path))
        if masks.ndim != 2 or tuple(masks.shape) != image_shape:
            raise ValueError(
                f"Dimension mismatch for {image_id}: image={image_shape}, mask={masks.shape}"
            )
        if not np.issubdtype(masks.dtype, np.integer) or np.any(masks < 0):
            raise ValueError(f"{mask_path.name} must contain non-negative integer labels")
        channels = tuple(int(value) for value in payload.get("chan_choose", ()))
        if channels != expected_channels:
            raise ValueError(
                f"{mask_path.name} uses channels {channels}, expected {expected_channels}"
            )
        instances = int(np.count_nonzero(np.unique(masks) > 0))
        if instances < 1:
            raise ValueError(f"{mask_path.name} contains no labeled instances")
        total_instances += instances
        (staging_directory / image_path.name).symlink_to(image_path.resolve())
        (staging_directory / mask_path.name).symlink_to(mask_path.resolve())

    return len(mask_paths), total_instances


def parse_args() -> argparse.Namespace:
    """Parse source, staging, and expected GUI-channel parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--chan", type=int, required=True)
    parser.add_argument("--chan2", type=int, required=True)
    parser.add_argument(
        "--image-list",
        type=Path,
        help="Optional text file containing one image ID per line",
    )
    return parser.parse_args()


def main() -> None:
    """Create a validated, symlink-only training snapshot without changing originals."""
    args = parse_args()
    image_ids = None
    if args.image_list is not None:
        image_ids = {
            line.strip()
            for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not image_ids:
            raise ValueError(f"No image IDs found in {args.image_list}")
    pairs, instances = stage_pairs(
        args.input_dir, args.staging_dir, (args.chan, args.chan2), image_ids
    )
    print(f"Staged {pairs} image-mask pairs containing {instances} instances")


if __name__ == "__main__":
    main()
