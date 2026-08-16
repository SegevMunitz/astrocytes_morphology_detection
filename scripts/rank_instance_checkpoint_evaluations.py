"""Rank fully evaluated instance checkpoints by PQ and F1, never by pixel loss."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def rank_evaluations(evaluation_root: Path, output_directory: Path) -> pd.DataFrame:
    """Aggregate IoU-0.5 tuning metrics and write an auditable best-checkpoint pointer."""
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(evaluation_root.rglob("evaluation.json")):
        metrics_path = metadata_path.parent / "metrics_by_image.csv"
        if not metrics_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = pd.read_csv(metrics_path)
        selected = metrics.loc[metrics["iou_threshold"] == 0.5]
        if "threshold_selection" in selected:
            tuning = selected.loc[selected["threshold_selection"] == "tuning_subset"]
            if not tuning.empty:
                selected = tuning
        if selected.empty:
            continue
        records.append(
            {
                "evaluation_directory": str(metadata_path.parent.resolve()),
                "run_directory": str(metadata.get("run_directory", "")),
                "checkpoint_name": str(metadata.get("checkpoint_name", "best.pt")),
                "image_count": int(selected["image_id"].nunique()),
                "f1": float(selected["f1"].mean()),
                "panoptic_quality": float(selected["panoptic_quality"].mean()),
                "precision": float(selected["precision"].mean()),
                "recall": float(selected["recall"].mean()),
            }
        )
    if not records:
        raise ValueError(f"No complete instance evaluations found below {evaluation_root}")
    ranking = pd.DataFrame(records).sort_values(
        ["panoptic_quality", "f1", "precision"], ascending=False
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_directory / "checkpoint_ranking.csv", index=False)
    best = ranking.iloc[0].to_dict()
    best["selection_metric"] = "mean validation panoptic_quality, then F1, at IoU 0.5"
    (output_directory / "best_checkpoint.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )
    return ranking


def main() -> None:
    """Parse evaluation root and ranking destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ranking = rank_evaluations(args.evaluation_root, args.output_dir)
    best = ranking.iloc[0]
    print(f"Best instance checkpoint: PQ={best.panoptic_quality:.4f}, F1={best.f1:.4f}")


if __name__ == "__main__":
    main()
