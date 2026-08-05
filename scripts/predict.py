"""Run overlap-averaged full-image inference and save probabilities, masks, and overlays."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from astroseg.datasets import prepare_model_inputs
from astroseg.inference import predict_full_image
from astroseg.io import get_channel, load_manifest, load_ome_tiff, load_yaml_configuration
from astroseg.models import build_model
from astroseg.training.checkpoints import load_checkpoint
from astroseg.visualization import save_segmentation_overlay


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load prediction configuration from a top-level YAML mapping.

    Model shape and patching settings are consumed from this mapping; invalid
    document roots fail before loading a potentially incompatible checkpoint.
    """
    return load_yaml_configuration(path)


def _resolve_path(value: str, manifest_path: Path, description: str) -> Path:
    """Resolve a manifest file path with context-specific error reporting.

    Both project-relative and manifest-relative layouts are supported, matching
    the same path convention used by datasets and preprocessing scripts.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} does not exist: {value}")


def _load_labels(path: Path) -> np.ndarray:
    """Load a nucleus instance-label image from NumPy or TIFF.

    The shared input-preparation function later checks dimensions and label values,
    keeping this helper limited to storage-format dispatch.
    """
    return np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else tifffile.imread(path)


def predict_split(
    configuration: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_directory: Path,
) -> int:
    """Predict every manifest image in one split at its original resolution.

    The function writes class probabilities, hard TIFF masks, and GFAP overlays
    to separate subdirectories and returns the number of processed images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = configuration["model"]
    model = build_model(
        model_config["architecture"],
        int(model_config["input_channels"]),
        int(model_config["num_classes"]),
        int(model_config.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    data_config = configuration["data"]
    manifest_path = Path(data_config["manifest_path"])
    manifest = load_manifest(manifest_path)
    rows = manifest.loc[manifest["split"] == split]
    if rows.empty:
        raise ValueError(f"Manifest contains no rows for split {split!r}")
    probability_directory = output_directory / "probabilities"
    mask_directory = output_directory / "masks"
    overlay_directory = output_directory / "overlays"
    for directory in (probability_directory, mask_directory, overlay_directory):
        directory.mkdir(parents=True, exist_ok=True)
    for _, row in rows.iterrows():
        if not row["cellpose_mask_path"].strip() or not row["gfap_channel"].strip():
            raise ValueError(f"Explicit nucleus-mask path and GFAP channel are required for {row['image_id']!r}")
        microscopy = load_ome_tiff(_resolve_path(row["path"], manifest_path, "Image"))
        labels = _load_labels(_resolve_path(row["cellpose_mask_path"], manifest_path, "Nucleus mask"))
        inputs = prepare_model_inputs(
            microscopy,
            row["gfap_channel"],
            labels,
            float(data_config.get("max_nucleus_distance", 64.0)),
        )
        probabilities = predict_full_image(
            model,
            inputs,
            int(data_config["patch_size"]),
            int(data_config["overlap"]),
            device,
        )
        class_mask = probabilities.argmax(axis=0).astype(np.uint8)
        np.save(probability_directory / f"{row['image_id']}.npy", probabilities, allow_pickle=False)
        tifffile.imwrite(mask_directory / f"{row['image_id']}.tiff", class_mask)
        gfap = get_channel(microscopy, row["gfap_channel"])
        save_segmentation_overlay(
            gfap,
            class_mask > 0,
            overlay_directory / f"{row['image_id']}.png",
            title="Prediction overlay",
        )
    return len(rows)


def parse_args() -> argparse.Namespace:
    """Parse model configuration, checkpoint, split, and output arguments.

    Test prediction and the standard prediction output directory are defaults,
    while configuration and checkpoint paths remain explicit requirements.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    return parser.parse_args()


def main() -> None:
    """Run split-level full-image inference and print a concise artifact summary.

    Any source-image, channel, nucleus-mask, or checkpoint mismatch propagates
    before the command claims successful completion.
    """
    args = parse_args()
    count = predict_split(_load_yaml(args.config), args.checkpoint, args.split, args.output_dir)
    print(f"Saved full-image predictions for {count} images to {args.output_dir}")


if __name__ == "__main__":
    main()
