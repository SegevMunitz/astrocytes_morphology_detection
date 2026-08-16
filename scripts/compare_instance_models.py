"""Compare custom and Cellpose instance metrics on identical validation images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compare_models(
    custom_directory: Path,
    cellpose_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Write per-image deltas and a concise mean-score comparison."""
    custom = pd.read_csv(custom_directory / "metrics_by_image.csv")
    custom = custom.loc[custom["iou_threshold"] == 0.5].copy()
    cellpose_metadata = json.loads(
        (cellpose_directory / "evaluation.json").read_text(encoding="utf-8")
    )
    cellpose = pd.read_csv(cellpose_directory / "threshold_metrics.csv")
    selected = cellpose.loc[
        (cellpose["flow_threshold"] == cellpose_metadata["best_flow_threshold"])
        & (
            cellpose["cellprob_threshold"]
            == cellpose_metadata["best_cellprob_threshold"]
        )
    ].copy()
    expected_ids = set(cellpose_metadata["image_ids"])
    custom = custom.loc[custom["image_id"].isin(expected_ids)]
    if set(custom["image_id"]) != expected_ids or set(selected["image_id"]) != expected_ids:
        raise ValueError("Both evaluations must contain exactly the Cellpose validation IDs")
    metrics = ["precision", "recall", "f1", "mean_matched_iou", "panoptic_quality"]
    merged = custom[["image_id", *metrics]].merge(
        selected[["image_id", *metrics]],
        on="image_id",
        suffixes=("_multichannel", "_cellpose"),
        validate="one_to_one",
    )
    for metric in metrics:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_multichannel"] - merged[f"{metric}_cellpose"]
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_directory / "metrics_by_image_comparison.csv", index=False)
    summary: dict[str, object] = {
        "comparison_scope": "identical explicit Cellpose validation images at IoU 0.5",
        "image_ids": sorted(expected_ids),
        "cellpose_thresholds": {
            "flow": cellpose_metadata["best_flow_threshold"],
            "cell_probability": cellpose_metadata["best_cellprob_threshold"],
        },
    }
    for metric in metrics:
        custom_mean = float(merged[f"{metric}_multichannel"].mean())
        cellpose_mean = float(merged[f"{metric}_cellpose"].mean())
        summary[metric] = {
            "multichannel": custom_mean,
            "cellpose": cellpose_mean,
            "delta": custom_mean - cellpose_mean,
        }
    summary["beats_cellpose_on_f1_and_pq"] = bool(
        summary["f1"]["delta"] > 0 and summary["panoptic_quality"]["delta"] > 0
    )
    (output_directory / "comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Parse evaluation directories and print the two decisive score deltas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-dir", type=Path, required=True)
    parser.add_argument("--cellpose-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = compare_models(args.custom_dir, args.cellpose_dir, args.output_dir)
    print(
        f"F1 delta={summary['f1']['delta']:+.4f}; "
        f"PQ delta={summary['panoptic_quality']['delta']:+.4f}; "
        f"beats_cellpose={summary['beats_cellpose_on_f1_and_pq']}"
    )


if __name__ == "__main__":
    main()
