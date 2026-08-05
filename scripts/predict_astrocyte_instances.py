"""Predict complete individual astrocytes with a trained nucleus-guided model."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch

from astroseg.analysis import astrocyte_instance_morphology
from astroseg.datasets import prepare_model_inputs
from astroseg.inference import predict_instance_full_image
from astroseg.io import (
    get_channel,
    load_manifest,
    load_ome_tiff,
    load_yaml_configuration,
    validate_manifest,
)
from astroseg.models import build_model
from astroseg.postprocessing import separate_astrocyte_instances
from astroseg.training.checkpoints import load_checkpoint
from astroseg.visualization import save_compartment_overlay, save_instance_overlay


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping used to reconstruct model and postprocessing settings.

    Invalid roots fail before loading large image data or allocating a model.
    """
    return load_yaml_configuration(path)


def _resolve_path(value: str, manifest_path: Path, description: str) -> Path:
    """Resolve project- or manifest-relative input paths without filename guessing.

    The supplied description keeps failures specific to image versus nucleus data.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} does not exist: {value}")


def _portable_path(path: Path) -> str:
    """Return a project-relative forward-slash path whenever possible.

    This keeps generated prediction manifests portable across Windows and POSIX.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_labels(path: Path) -> np.ndarray:
    """Load a two-dimensional nucleus label array from NumPy or TIFF storage.

    Full alignment and label validation is performed by input preparation.
    """
    return np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else tifffile.imread(path)


def predict_split(
    configuration: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_directory: Path,
    output_manifest: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Predict raw heads and final instances for every image in one manifest split.

    Automatic paths are written to dedicated ``predicted_*`` columns. Human
    annotation columns remain untouched for correction and evaluation provenance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_configuration = configuration["model"]
    model = build_model(
        str(model_configuration["architecture"]),
        int(model_configuration["input_channels"]),
        int(model_configuration["num_classes"]),
        int(model_configuration.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    data = configuration["data"]
    manifest_path = Path(data["manifest_path"])
    manifest = load_manifest(manifest_path)
    rows = manifest.loc[manifest["split"] == split]
    if rows.empty:
        raise ValueError(f"Manifest contains no rows for split {split!r}")
    heads_directory = output_directory / "raw_heads"
    labels_directory = output_directory / "labels"
    compartments_directory = output_directory / "compartments"
    overlays_directory = output_directory / "overlays"
    measurements: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    postprocessing = configuration.get("postprocessing", {})
    for index, row in rows.iterrows():
        image_id = str(row["image_id"])
        microscopy = load_ome_tiff(_resolve_path(str(row["path"]), manifest_path, "Image"))
        nuclei = _load_labels(
            _resolve_path(str(row["cellpose_mask_path"]), manifest_path, "Nucleus labels")
        )
        inputs = prepare_model_inputs(
            microscopy,
            str(row["gfap_channel"]),
            nuclei,
            float(data.get("max_nucleus_distance", 64)),
        )
        heads = predict_instance_full_image(
            model,
            inputs,
            int(data["patch_size"]),
            int(data["overlap"]),
            device,
        )
        result = separate_astrocyte_instances(
            1.0 - heads.semantic_probabilities[0],
            nuclei,
            semantic_probabilities=heads.semantic_probabilities,
            boundary_probability=heads.boundary_probability,
            ownership_offsets=heads.ownership_offsets,
            foreground_threshold=float(postprocessing.get("foreground_threshold", 0.5)),
            boundary_threshold=float(postprocessing.get("boundary_threshold", 0.5)),
            max_nucleus_to_gfap_distance=float(
                postprocessing.get("max_nucleus_to_gfap_distance", 16)
            ),
            offset_scale=float(postprocessing.get("offset_scale", data.get("offset_scale", 256))),
            max_offset_endpoint_distance=float(
                postprocessing.get("max_offset_endpoint_distance", 32)
            ),
            soma_expansion=int(postprocessing.get("soma_expansion", 4)),
            soma_radius=float(postprocessing.get("soma_radius", data.get("soma_radius", 20))),
            min_cell_area=int(postprocessing.get("min_cell_area", 50)),
            min_gfap_area=int(postprocessing.get("min_gfap_area", 20)),
        )
        head_path = heads_directory / f"{image_id}.npz"
        label_path = labels_directory / f"{image_id}.tiff"
        compartment_path = compartments_directory / f"{image_id}.tiff"
        instance_overlay = overlays_directory / f"{image_id}_instances.png"
        compartment_overlay = overlays_directory / f"{image_id}_compartments.png"
        destinations = (head_path, label_path, compartment_path, instance_overlay, compartment_overlay)
        existing = [path for path in destinations if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"Refusing to overwrite prediction artifacts: {existing}")
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            head_path,
            semantic_probabilities=heads.semantic_probabilities,
            boundary_probability=heads.boundary_probability,
            ownership_offsets=heads.ownership_offsets,
        )
        tifffile.imwrite(label_path, result.labels, photometric="minisblack")
        tifffile.imwrite(compartment_path, result.compartments, photometric="minisblack")
        gfap = get_channel(microscopy, str(row["gfap_channel"]))
        save_instance_overlay(gfap, result.labels, instance_overlay)
        save_compartment_overlay(gfap, result.compartments, compartment_overlay)
        frame = astrocyte_instance_morphology(
            result.labels, result.compartments, result.cell_to_nucleus
        )
        frame.insert(0, "image_id", image_id)
        measurements.append(frame)
        manifest.loc[index, "predicted_instance_path"] = _portable_path(label_path)
        manifest.loc[index, "predicted_compartment_path"] = _portable_path(compartment_path)
        manifest.loc[index, "instance_prediction_status"] = "automatic"
        manifest.loc[index, "instance_prediction_source"] = "nucleus_guided_instance_unet"
        reports.append(
            {
                "image_id": image_id,
                "cell_count": result.cell_count,
                "active_nucleus_count": result.active_nucleus_count,
                "unassigned_foreground_fraction": result.unassigned_foreground_fraction,
                "ownership_mode": result.ownership_mode,
            }
        )
    validate_manifest(manifest)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    pd.concat(measurements, ignore_index=True).to_csv(
        output_directory / "cell_measurements.csv", index=False
    )
    pd.DataFrame(reports).to_csv(output_directory / "prediction_report.csv", index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse trained checkpoint, split, output, and overwrite options.

    The instance-specific YAML is the default while the trained checkpoint remains
    explicit to prevent accidental use of a binary model.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_instances.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/instance_predictions"))
    parser.add_argument(
        "--output-manifest", type=Path, default=Path("outputs/instance_predictions/manifest.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate complete-cell predictions without altering human annotation fields.

    A concise image count is printed only after every artifact and manifest writes.
    """
    args = parse_args()
    manifest = predict_split(
        _load_yaml(args.config),
        args.checkpoint,
        args.split,
        args.output_dir,
        args.output_manifest,
        args.overwrite,
    )
    print(f"Saved trained instance predictions for {len(manifest)} manifest image(s)")


if __name__ == "__main__":
    main()
