"""Generate automatic binary labels for unannotated images and update a new manifest."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch

from astroseg.annotations import save_pseudo_label_artifacts
from astroseg.datasets import prepare_model_inputs
from astroseg.inference import predict_full_image
from astroseg.io import (
    get_channel,
    load_manifest,
    load_ome_tiff,
    load_yaml_configuration,
    validate_manifest,
)
from astroseg.models import build_model
from astroseg.training.checkpoints import load_checkpoint


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load pseudo-label model and data settings from a YAML mapping.

    The configuration path must exist and its root must be a dictionary because
    nested data and model sections are consumed without implicit defaults.
    """
    return load_yaml_configuration(path)


def _resolve_path(value: str, manifest_path: Path, description: str) -> Path:
    """Resolve a manifest file reference and preserve context in failures.

    Project-relative paths are preferred, with manifest-relative paths supported
    for portable pair tables. The description distinguishes images from masks.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} does not exist: {value}")


def _portable_path(path: Path) -> str:
    """Represent an output path relative to the project when possible.

    Portable forward-slash paths are written to generated manifests; paths outside
    the working tree remain explicit rather than being rewritten incorrectly.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_labels(path: Path) -> np.ndarray:
    """Load nucleus instance labels from NumPy or TIFF for model-input creation.

    No binarization occurs here because ``prepare_model_inputs`` performs and
    validates the shared conversion used during training and ordinary prediction.
    """
    labels = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else tifffile.imread(path)
    return np.asarray(labels)


def generate_pseudo_labels(
    configuration: dict[str, Any],
    checkpoint_path: Path,
    output_directory: Path,
    output_manifest: Path,
    split: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate automatic labels only for manifest rows in state ``none``.

    Full-image probabilities, hard masks, and overlays remain under the automatic
    output directory. A new manifest records ``pseudo`` provenance for later review.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = configuration["model"]
    if int(model_config["num_classes"]) != 2:
        raise ValueError("Pseudo-label generation currently supports binary segmentation only")
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
    selected = manifest["annotation_status"] == "none"
    if split is not None:
        selected &= manifest["split"] == split
    indices = manifest.index[selected].tolist()
    if not indices:
        raise ValueError("No unannotated manifest rows match the requested split")
    for index in indices:
        row = manifest.loc[index]
        if not str(row["cellpose_mask_path"]).strip() or not str(row["gfap_channel"]).strip():
            raise ValueError(
                f"Explicit nucleus-mask path and GFAP channel are required for {row['image_id']!r}"
            )
        microscopy = load_ome_tiff(_resolve_path(str(row["path"]), manifest_path, "Image"))
        labels = _load_labels(
            _resolve_path(str(row["cellpose_mask_path"]), manifest_path, "Nucleus mask")
        )
        inputs = prepare_model_inputs(
            microscopy,
            str(row["gfap_channel"]),
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
        gfap = get_channel(microscopy, str(row["gfap_channel"]))
        artifacts = save_pseudo_label_artifacts(
            str(row["image_id"]), probabilities, gfap, output_directory, overwrite
        )
        manifest.loc[index, "annotation_path"] = _portable_path(artifacts.mask_path)
        manifest.loc[index, "annotation_status"] = "pseudo"
        manifest.loc[index, "annotation_source"] = f"model:{_portable_path(checkpoint_path)}"
        manifest.loc[index, "annotator"] = ""
        manifest.loc[index, "review_status"] = "pending"
    validate_manifest(manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, configuration, output, and optional split arguments.

    A destination manifest is required to avoid silently replacing the input
    catalog. Overwriting automatic files requires an explicit flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pseudo_labels"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run pseudo-label inference and summarize the resulting manifest states.

    The printed count reflects all rows marked pseudo in the output manifest,
    while detailed errors remain associated with their source image IDs.
    """
    args = parse_args()
    manifest = generate_pseudo_labels(
        _load_yaml(args.config),
        args.checkpoint,
        args.output_dir,
        args.output_manifest,
        args.split,
        args.overwrite,
    )
    count = int((manifest["annotation_status"] == "pseudo").sum())
    print(f"Recorded {count} pseudo-labeled images in {args.output_manifest}")


if __name__ == "__main__":
    main()
