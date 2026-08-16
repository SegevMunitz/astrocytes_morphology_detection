"""Evaluate held-out folds and tune instance reconstruction thresholds."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch

from astroseg.datasets import prepare_model_inputs
from astroseg.inference import InstanceHeadPredictions, predict_instance_full_image
from astroseg.io import load_ome_tiff
from astroseg.models import build_model
from astroseg.postprocessing import AstrocyteInstanceResult, separate_astrocyte_instances
from astroseg.training import instance_segmentation_metrics
from astroseg.training.checkpoints import load_checkpoint


def _load_labels(path: str | Path) -> np.ndarray:
    """Load a plain TIFF/NumPy label image from an explicit manifest path."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return (
        np.load(source, allow_pickle=False)
        if source.suffix.lower() == ".npy"
        else tifffile.imread(source)
    )


def _read_ids(path: Path | None) -> set[str]:
    """Read an optional comment-aware list used only for threshold selection."""
    if path is None:
        return set()
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"No image IDs in {path}")
    return values


def _restore_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Restore one fold model and the exact training configuration it records."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    configuration = checkpoint["configuration"]
    model_configuration = configuration["model"]
    model = build_model(
        str(model_configuration["architecture"]),
        int(model_configuration["input_channels"]),
        int(model_configuration["num_classes"]),
        int(model_configuration.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, configuration


def _restore_configuration(checkpoint_path: Path) -> dict[str, Any]:
    """Load only the recorded configuration when raw predictions are reused."""
    return load_checkpoint(checkpoint_path, torch.device("cpu"))["configuration"]


def _predict_row(
    row: pd.Series,
    model: torch.nn.Module,
    configuration: dict[str, Any],
    device: torch.device,
) -> tuple[InstanceHeadPredictions, np.ndarray]:
    """Predict raw full-resolution heads for one fold's held-out image."""
    microscopy = load_ome_tiff(str(row["path"]))
    nuclei = _load_labels(str(row["cellpose_mask_path"]))
    data = configuration["data"]
    inputs = prepare_model_inputs(
        microscopy,
        str(row["gfap_channel"]),
        nuclei,
        float(data.get("max_nucleus_distance", 64)),
        input_mode=str(data.get("input_mode", "nucleus_guidance")),
        auxiliary_channel=str(row.get("auxiliary_channel", "")),
        dapi_channel=str(row.get("dapi_channel", "")),
    )
    heads = predict_instance_full_image(
        model,
        inputs,
        int(data["patch_size"]),
        int(data["overlap"]),
        device,
    )
    return heads, nuclei


def _separate(
    heads: InstanceHeadPredictions,
    nuclei: np.ndarray,
    configuration: dict[str, Any],
    foreground_threshold: float,
    boundary_threshold: float,
    max_nucleus_distance: float | None = None,
    min_nucleus_foreground_fraction: float | None = None,
    nucleus_support_expansion: int | None = None,
) -> AstrocyteInstanceResult:
    """Apply a candidate threshold pair while retaining configured geometry."""
    data = configuration["data"]
    postprocessing = configuration.get("postprocessing", {})
    return separate_astrocyte_instances(
        1.0 - heads.semantic_probabilities[0],
        nuclei,
        semantic_probabilities=heads.semantic_probabilities,
        boundary_probability=heads.boundary_probability,
        ownership_offsets=heads.ownership_offsets,
        foreground_threshold=foreground_threshold,
        boundary_threshold=boundary_threshold,
        max_nucleus_to_gfap_distance=float(
            max_nucleus_distance
            if max_nucleus_distance is not None
            else postprocessing.get("max_nucleus_to_gfap_distance", 16)
        ),
        nucleus_support_expansion=int(
            nucleus_support_expansion
            if nucleus_support_expansion is not None
            else postprocessing.get("nucleus_support_expansion", 4)
        ),
        min_nucleus_foreground_fraction=float(
            min_nucleus_foreground_fraction
            if min_nucleus_foreground_fraction is not None
            else postprocessing.get("min_nucleus_foreground_fraction", 0)
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


def _save_heads(path: Path, heads: InstanceHeadPredictions) -> None:
    """Persist raw heads so postprocessing is reproducible without another GPU pass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        semantic_probabilities=heads.semantic_probabilities,
        boundary_probability=heads.boundary_probability,
        ownership_offsets=heads.ownership_offsets,
    )


def _load_heads(path: Path) -> InstanceHeadPredictions:
    """Restore raw heads written by :func:`_save_heads`."""
    with np.load(path, allow_pickle=False) as payload:
        return InstanceHeadPredictions(
            semantic_probabilities=payload["semantic_probabilities"],
            boundary_probability=payload["boundary_probability"],
            ownership_offsets=payload["ownership_offsets"],
        )


def evaluate_cross_validation(
    run_directory: Path,
    output_directory: Path,
    tuning_image_ids: set[str] | None = None,
    *,
    raw_heads_directory: Path | None = None,
    foreground_thresholds: Sequence[float] = (0.35, 0.45, 0.55, 0.65),
    boundary_thresholds: Sequence[float] = (0.35, 0.50, 0.65),
    max_nucleus_distances: Sequence[float] = (8.0, 16.0, 32.0),
    min_nucleus_foreground_fractions: Sequence[float] = (0.0,),
    nucleus_support_expansion: int = 4,
    checkpoint_name: str = "best.pt",
) -> pd.DataFrame:
    """Generate OOF heads, tune thresholds, and aggregate object-level metrics.

    If ``tuning_image_ids`` is supplied, only those held-out images select the
    threshold pair. Scores for every OOF image are still retained so the tuning
    scope and the remaining held-out performance cannot be conflated.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_directory.mkdir(parents=True, exist_ok=True)
    candidate_records: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    threshold_settings = list(
        itertools.product(
            foreground_thresholds,
            boundary_thresholds,
            max_nucleus_distances,
            min_nucleus_foreground_fractions,
        )
    )
    if not threshold_settings:
        raise ValueError("Postprocessing threshold grid cannot be empty")
    if any(not 0.0 <= value <= 1.0 for value in foreground_thresholds):
        raise ValueError("foreground_thresholds must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in boundary_thresholds):
        raise ValueError("boundary_thresholds must be in [0, 1]")
    if any(value <= 0 for value in max_nucleus_distances):
        raise ValueError("max_nucleus_distances must be positive")
    if any(not 0.0 <= value <= 1.0 for value in min_nucleus_foreground_fractions):
        raise ValueError("min_nucleus_foreground_fractions must be in [0, 1]")
    if nucleus_support_expansion < 0:
        raise ValueError("nucleus_support_expansion must be non-negative")
    for fold_directory in sorted(run_directory.glob("fold_*")):
        assignments_path = fold_directory / "cross_validation_assignments.csv"
        checkpoint_path = fold_directory / checkpoint_name
        if not assignments_path.is_file() or not checkpoint_path.is_file():
            continue
        model = None
        if raw_heads_directory is None:
            model, configuration = _restore_model(checkpoint_path, device)
        else:
            configuration = _restore_configuration(checkpoint_path)
        assignments = pd.read_csv(assignments_path, dtype=str, keep_default_na=False)
        held_out = assignments.loc[assignments["split"] == "val"]
        for _, row in held_out.iterrows():
            image_id = str(row["image_id"])
            nuclei = _load_labels(str(row["cellpose_mask_path"]))
            if raw_heads_directory is None:
                assert model is not None
                heads, predicted_nuclei = _predict_row(
                    row, model, configuration, device
                )
                if not np.array_equal(nuclei, predicted_nuclei):
                    raise ValueError(f"Nucleus inputs changed for {image_id!r}")
                head_path = output_directory / "raw_heads" / f"{image_id}.npz"
                _save_heads(head_path, heads)
            else:
                head_path = raw_heads_directory / f"{image_id}.npz"
                if not head_path.is_file():
                    raise FileNotFoundError(
                        f"Saved raw heads do not exist for {image_id!r}: {head_path}"
                    )
                heads = _load_heads(head_path)
            target = _load_labels(str(row["instance_annotation_path"]))
            examples.append(
                {
                    "fold": fold_directory.name,
                    "image_id": image_id,
                    "head_path": head_path,
                    "nucleus_path": Path(str(row["cellpose_mask_path"])),
                    "target_path": Path(str(row["instance_annotation_path"])),
                    "configuration": configuration,
                }
            )
            for (
                foreground_threshold,
                boundary_threshold,
                max_nucleus_distance,
                min_nucleus_foreground_fraction,
            ) in threshold_settings:
                error = ""
                try:
                    separated = _separate(
                        heads,
                        nuclei,
                        configuration,
                        foreground_threshold,
                        boundary_threshold,
                        max_nucleus_distance,
                        min_nucleus_foreground_fraction,
                        nucleus_support_expansion,
                    )
                    prediction = separated.labels
                except ValueError as exception:
                    prediction = np.zeros(target.shape, dtype=np.uint32)
                    error = str(exception)
                metrics = instance_segmentation_metrics(prediction, target, 0.5)
                metrics.pop("matches")
                candidate_records.append(
                    {
                        "fold": fold_directory.name,
                        "image_id": image_id,
                        "foreground_threshold": foreground_threshold,
                        "boundary_threshold": boundary_threshold,
                        "max_nucleus_to_gfap_distance": max_nucleus_distance,
                        "min_nucleus_foreground_fraction": (
                            min_nucleus_foreground_fraction
                        ),
                        "nucleus_support_expansion": nucleus_support_expansion,
                        **metrics,
                        "postprocessing_error": error,
                    }
                )
        if model is not None:
            del model
        if raw_heads_directory is None and device.type == "cuda":
            torch.cuda.empty_cache()
    if not examples:
        raise ValueError(f"No complete held-out folds found in {run_directory}")

    candidates = pd.DataFrame(candidate_records)
    candidates.to_csv(output_directory / "threshold_metrics.csv", index=False)
    available_ids = set(candidates["image_id"])
    requested_ids = set(tuning_image_ids or ())
    unknown_ids = requested_ids - available_ids
    if unknown_ids:
        raise ValueError(f"Tuning IDs are absent from OOF predictions: {sorted(unknown_ids)}")
    selection_ids = requested_ids or available_ids
    ranking = (
        candidates.loc[candidates["image_id"].isin(selection_ids)]
        .groupby(
            [
                "foreground_threshold",
                "boundary_threshold",
                "max_nucleus_to_gfap_distance",
                "min_nucleus_foreground_fraction",
                "nucleus_support_expansion",
            ]
        )[
            ["f1", "panoptic_quality", "precision", "recall"]
        ]
        .mean()
        .reset_index()
        .sort_values(["panoptic_quality", "f1"], ascending=False)
    )
    ranking.to_csv(output_directory / "threshold_ranking.csv", index=False)
    best = ranking.iloc[0]
    best_foreground = float(best["foreground_threshold"])
    best_boundary = float(best["boundary_threshold"])
    best_nucleus_distance = float(best["max_nucleus_to_gfap_distance"])
    best_nucleus_fraction = float(best["min_nucleus_foreground_fraction"])
    best_support_expansion = int(best["nucleus_support_expansion"])

    records: list[dict[str, Any]] = []
    for example in examples:
        heads = _load_heads(example["head_path"])
        nuclei = _load_labels(example["nucleus_path"])
        target = _load_labels(example["target_path"])
        separated = _separate(
            heads,
            nuclei,
            example["configuration"],
            best_foreground,
            best_boundary,
            best_nucleus_distance,
            best_nucleus_fraction,
            best_support_expansion,
        )
        prediction_path = output_directory / "labels" / f"{example['image_id']}.tiff"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(prediction_path, separated.labels, photometric="minisblack")
        for threshold in (0.5, 0.75):
            metrics = instance_segmentation_metrics(separated.labels, target, threshold)
            metrics.pop("matches")
            records.append(
                {
                    "fold": example["fold"],
                    "image_id": example["image_id"],
                    "threshold_selection": (
                        "tuning_subset" if example["image_id"] in selection_ids else "evaluation_only"
                    ),
                    "iou_threshold": threshold,
                    **metrics,
                    "predicted_count": separated.cell_count,
                    "active_nucleus_count": separated.active_nucleus_count,
                    "rejected_nucleus_count": separated.rejected_nucleus_count,
                    "unassigned_foreground_fraction": separated.unassigned_foreground_fraction,
                    "prediction_path": str(prediction_path.resolve()),
                }
            )
    result = pd.DataFrame(records)
    result.to_csv(output_directory / "metrics_by_image.csv", index=False)
    metric_columns = [
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "panoptic_quality",
        "unassigned_foreground_fraction",
    ]
    summary = result.groupby("iou_threshold")[metric_columns].agg(["mean", "std"])
    summary.to_csv(output_directory / "metrics_summary.csv")
    metadata = {
        "run_directory": str(run_directory.resolve()),
        "checkpoint_name": checkpoint_name,
        "held_out_images": sorted(available_ids),
        "folds": sorted(result["fold"].unique().tolist()),
        "device": str(device),
        "threshold_selection_ids": sorted(selection_ids),
        "threshold_selection_scope": (
            "explicit held-out tuning subset" if requested_ids else "all OOF images"
        ),
        "best_foreground_threshold": best_foreground,
        "best_boundary_threshold": best_boundary,
        "best_max_nucleus_to_gfap_distance": best_nucleus_distance,
        "best_min_nucleus_foreground_fraction": best_nucleus_fraction,
        "best_nucleus_support_expansion": best_support_expansion,
        "best_tuning_mean_f1": float(best["f1"]),
        "best_tuning_mean_panoptic_quality": float(best["panoptic_quality"]),
    }
    (output_directory / "evaluation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _parse_float_values(value: str) -> tuple[float, ...]:
    """Parse a non-empty comma-separated finite threshold list."""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exception:
        raise argparse.ArgumentTypeError(f"Invalid numeric list: {value!r}") from exception
    if not values or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("Threshold lists must be non-empty and finite")
    return values


def parse_args() -> argparse.Namespace:
    """Parse completed fold directory, tuning subset, and output destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tuning-image-list", type=Path)
    parser.add_argument(
        "--raw-heads-dir",
        type=Path,
        help="Reuse saved full-image heads instead of running the model again",
    )
    parser.add_argument(
        "--foreground-thresholds",
        type=_parse_float_values,
        default=(0.35, 0.45, 0.55, 0.65),
    )
    parser.add_argument(
        "--boundary-thresholds",
        type=_parse_float_values,
        default=(0.35, 0.50, 0.65),
    )
    parser.add_argument(
        "--max-nucleus-distances",
        type=_parse_float_values,
        default=(8.0, 16.0, 32.0),
    )
    parser.add_argument(
        "--min-nucleus-foreground-fractions",
        type=_parse_float_values,
        default=(0.0,),
    )
    parser.add_argument("--nucleus-support-expansion", type=int, default=4)
    parser.add_argument("--checkpoint-name", default="best.pt")
    return parser.parse_args()


def main() -> None:
    """Evaluate all available folds and print the IoU-0.5 mean scores."""
    args = parse_args()
    result = evaluate_cross_validation(
        args.run_dir,
        args.output_dir,
        _read_ids(args.tuning_image_list),
        raw_heads_directory=args.raw_heads_dir,
        foreground_thresholds=args.foreground_thresholds,
        boundary_thresholds=args.boundary_thresholds,
        max_nucleus_distances=args.max_nucleus_distances,
        min_nucleus_foreground_fractions=(
            args.min_nucleus_foreground_fractions
        ),
        nucleus_support_expansion=args.nucleus_support_expansion,
        checkpoint_name=args.checkpoint_name,
    )
    mean = result.loc[result["iou_threshold"] == 0.5, ["f1", "panoptic_quality"]].mean()
    print(
        f"Held-out images={result.image_id.nunique()} "
        f"F1={mean.f1:.4f} PQ={mean.panoptic_quality:.4f}"
    )


if __name__ == "__main__":
    main()
