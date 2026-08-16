"""Select a Cellpose checkpoint by validation PQ/F1 after threshold tuning."""

import argparse
import json
from pathlib import Path

import pandas as pd


def rank_cellpose_evaluations(evaluation_root: Path, output: Path) -> pd.DataFrame:
    """Rank completed evaluations by PQ, F1, and checkpoint path."""
    records = []
    for path in sorted(evaluation_root.rglob("evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "best_mean_panoptic_quality" not in payload:
            continue
        records.append(
            {
                "checkpoint": payload["checkpoint"],
                "evaluation_directory": str(path.parent.resolve()),
                "input_channels": payload.get("input_channels", 2),
                "zero_auxiliary": payload.get("zero_auxiliary", False),
                "panoptic_quality": payload["best_mean_panoptic_quality"],
                "f1": payload["best_mean_f1"],
                "flow_threshold": payload["best_flow_threshold"],
                "cellprob_threshold": payload["best_cellprob_threshold"],
            }
        )
    if not records:
        raise ValueError(f"No completed Cellpose evaluations found below {evaluation_root}")
    ranking = pd.DataFrame(records).sort_values(
        ["panoptic_quality", "f1"], ascending=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output, index=False)
    best = ranking.iloc[0].to_dict()
    best["selection_metric"] = "validation panoptic_quality, then F1, at IoU 0.5"
    output.with_suffix(".json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )
    return ranking


def main() -> None:
    """Parse evaluation tree and ranking output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranking = rank_cellpose_evaluations(args.evaluation_root, args.output)
    best = ranking.iloc[0]
    print(f"Best Cellpose checkpoint: PQ={best.panoptic_quality:.4f}, F1={best.f1:.4f}")


if __name__ == "__main__":
    main()
