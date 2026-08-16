"""Measure a trained Cellpose model on explicit held-out instance masks."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from cellpose import io, models

from astroseg.training import instance_segmentation_metrics


def _read_ids(path: Path) -> list[str]:
    """Read a non-empty comment-aware image-ID list."""
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError(f"No image IDs in {path}")
    return values


def _find_image(directory: Path, image_id: str) -> Path:
    """Find exactly one TIFF/BMP image for an explicit held-out ID."""
    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.stem == image_id
        and path.suffix.lower() in {".tif", ".tiff", ".bmp"}
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {image_id!r}; found {len(matches)}")
    return matches[0]


def evaluate_cellpose(
    image_directory: Path,
    mask_directory: Path,
    image_ids: list[str],
    checkpoint: Path,
    output_directory: Path,
    channels: tuple[int, int] = (1, 3),
) -> pd.DataFrame:
    """Tune standard Cellpose thresholds and report held-out object metrics.

    Threshold selection is deliberately reported as validation tuning, not as an
    unbiased test result. This gives the historical Cellpose baseline a strong,
    transparent comparison against the out-of-fold custom model.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Cellpose evaluation")
    images = [io.imread(str(_find_image(image_directory, image_id))) for image_id in image_ids]
    targets = []
    for image_id in image_ids:
        path = mask_directory / f"{image_id}_seg.npy"
        payload = np.load(path, allow_pickle=True).item()
        targets.append(np.asarray(payload["masks"]))
    model = models.CellposeModel(gpu=True, pretrained_model=str(checkpoint))
    records: list[dict[str, float | int | str]] = []
    predictions_by_setting: dict[tuple[float, float], list[np.ndarray]] = {}
    for flow_threshold, cell_probability in itertools.product(
        (0.4, 0.6, 0.8), (-1.0, 0.0, 1.0)
    ):
        predicted, _, _ = model.eval(
            images,
            channels=list(channels),
            diameter=None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cell_probability,
        )
        predicted_arrays = [np.asarray(value) for value in predicted]
        predictions_by_setting[(flow_threshold, cell_probability)] = predicted_arrays
        for image_id, prediction, target in zip(
            image_ids, predicted_arrays, targets, strict=True
        ):
            metrics = instance_segmentation_metrics(prediction, target, 0.5)
            metrics.pop("matches")
            records.append(
                {
                    "image_id": image_id,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cell_probability,
                    **metrics,
                }
            )
    result = pd.DataFrame(records)
    ranking = (
        result.groupby(["flow_threshold", "cellprob_threshold"])[
            ["f1", "panoptic_quality", "precision", "recall"]
        ]
        .mean()
        .reset_index()
        .sort_values(["panoptic_quality", "f1"], ascending=False)
    )
    best = ranking.iloc[0]
    setting = (float(best["flow_threshold"]), float(best["cellprob_threshold"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    labels_directory = output_directory / "labels"
    labels_directory.mkdir(exist_ok=True)
    for image_id, prediction in zip(
        image_ids, predictions_by_setting[setting], strict=True
    ):
        tifffile.imwrite(
            labels_directory / f"{image_id}.tiff",
            prediction.astype(np.uint32),
            photometric="minisblack",
        )
    result.to_csv(output_directory / "threshold_metrics.csv", index=False)
    ranking.to_csv(output_directory / "threshold_ranking.csv", index=False)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "image_ids": image_ids,
        "channels": list(channels),
        "selection_scope": "thresholds tuned on held-out validation masks",
        "best_flow_threshold": setting[0],
        "best_cellprob_threshold": setting[1],
        "best_mean_f1": float(best["f1"]),
        "best_mean_panoptic_quality": float(best["panoptic_quality"]),
    }
    (output_directory / "evaluation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    """Parse authoritative images/masks, validation IDs, and checkpoint path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chan", type=int, default=1)
    parser.add_argument("--chan2", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run the validation sweep and print the strongest Cellpose setting."""
    args = parse_args()
    result = evaluate_cellpose(
        args.image_dir,
        args.mask_dir,
        _read_ids(args.image_list),
        args.checkpoint,
        args.output_dir,
        (args.chan, args.chan2),
    )
    summary = result.groupby(["flow_threshold", "cellprob_threshold"])[
        ["f1", "panoptic_quality"]
    ].mean()
    best = summary.sort_values(["panoptic_quality", "f1"], ascending=False).iloc[0]
    print(f"Cellpose validation F1={best.f1:.4f} PQ={best.panoptic_quality:.4f}")


if __name__ == "__main__":
    main()
