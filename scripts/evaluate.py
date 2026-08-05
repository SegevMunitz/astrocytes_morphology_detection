"""Evaluate a checkpoint on a manifest split and save patch-aggregated image metrics."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from astroseg.datasets import AstrocyteDataset, collate_segmentation_batch
from astroseg.io import load_yaml_configuration
from astroseg.constants import TRAINABLE_ANNOTATION_STATUSES
from astroseg.models import build_model
from astroseg.training.checkpoints import load_checkpoint
from astroseg.training.metrics import metrics_from_probability_patches
from astroseg.training.cross_validation import load_grouped_fold_manifests


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one evaluation configuration as a validated top-level mapping.

    The helper preserves nested values for model and data construction and rejects
    non-mapping YAML documents before checkpoint or dataset access.
    """
    return load_yaml_configuration(path)


def evaluate_checkpoint(
    configuration: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_path: Path,
) -> pd.DataFrame:
    """Evaluate a checkpoint after reconstructing each complete source image.

    Patch probabilities are averaged in overlaps before metrics are computed once
    per pixel. The CSV contains per-image rows and an image-macro aggregate row.
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
    annotation_statuses = data_config.get(
        "train_annotation_statuses", sorted(TRAINABLE_ANNOTATION_STATUSES)
    )
    manifest_path = Path(data_config["manifest_path"])
    cross_validation = configuration.get("cross_validation", {})
    dataset_manifest: Path | pd.DataFrame
    dataset_split = split
    manifest_base_directory: Path | None = None
    if cross_validation.get("enabled", False) and split in {"train", "val"}:
        train_manifest, validation_manifest = load_grouped_fold_manifests(
            manifest_path,
            int(cross_validation.get("n_splits", 5)),
            int(cross_validation.get("validation_fold", 0)),
            str(cross_validation.get("group_column", "image_id")),
            str(cross_validation.get("fold_column", "fold")),
            int(configuration.get("seed", 42)),
            annotation_statuses,
        )
        dataset_manifest = train_manifest if split == "train" else validation_manifest
        manifest_base_directory = manifest_path.parent
    else:
        dataset_manifest = manifest_path
    dataset = AstrocyteDataset(
        dataset_manifest,
        dataset_split,
        int(data_config["patch_size"]),
        int(data_config["overlap"]),
        float(data_config.get("max_nucleus_distance", 64.0)),
        annotation_statuses=annotation_statuses,
        manifest_base_directory=manifest_base_directory,
        num_classes=int(model_config["num_classes"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(configuration["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        collate_fn=collate_segmentation_batch,
    )
    patches_by_image: dict[str, dict[str, list[Any]]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            probabilities = torch.softmax(model(batch["image"].to(device)), dim=1).cpu().numpy()
            for index, image_id in enumerate(batch["image_id"]):
                collection = patches_by_image.setdefault(
                    image_id, {"probabilities": [], "targets": [], "coordinates": []}
                )
                collection["probabilities"].append(probabilities[index])
                collection["targets"].append(batch["target"][index].numpy())
                collection["coordinates"].append(batch["coordinates"][index])
    records: list[dict[str, Any]] = []
    for image_id, collection in patches_by_image.items():
        coordinates = collection["coordinates"]
        image_shape = (
            max(coordinate.y + coordinate.height for coordinate in coordinates),
            max(coordinate.x + coordinate.width for coordinate in coordinates),
        )
        values = metrics_from_probability_patches(
            collection["probabilities"],
            collection["targets"],
            coordinates,
            image_shape,
        )
        record: dict[str, Any] = {"image_id": image_id}
        for class_index, class_values in values["per_class"].items():
            for name, value in class_values.items():
                record[f"class_{class_index}_{name}"] = value
        for name, value in values["macro"].items():
            record[f"macro_{name}"] = value
        records.append(record)
    frame = pd.DataFrame(records)
    aggregate = {"image_id": "__aggregate__"}
    aggregate.update(frame.drop(columns="image_id").mean().to_dict())
    frame = pd.concat((frame, pd.DataFrame([aggregate])), ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    """Parse checkpoint-evaluation command-line options.

    Configuration and checkpoint paths are required; split and output location
    receive reproducible defaults suitable for validation evaluation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/evaluation.csv"))
    return parser.parse_args()


def main() -> None:
    """Run configured evaluation and print a concise output summary.

    The command reports the number of reconstructed image rows separately from
    the final aggregate row saved in the metrics CSV.
    """
    args = parse_args()
    frame = evaluate_checkpoint(_load_yaml(args.config), args.checkpoint, args.split, args.output)
    print(f"Wrote metrics for {len(frame) - 1} images plus aggregate to {args.output}")


if __name__ == "__main__":
    main()
