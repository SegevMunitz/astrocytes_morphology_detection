"""Shortlist saved Cellpose snapshots for subsequent PQ/F1 evaluation."""

import argparse
import re
from pathlib import Path

import pandas as pd


EPOCH_PATTERN = re.compile(r"_epoch_(\d+)$")


def shortlist_candidates(run_root: Path, candidates_per_run: int = 2) -> pd.DataFrame:
    """Select low-loss snapshots only to reduce expensive object-metric evaluations."""
    if candidates_per_run < 1:
        raise ValueError("candidates_per_run must be positive")
    records: list[dict[str, object]] = []
    for history_path in sorted(run_root.glob("lr_*/result/history.csv")):
        history = pd.read_csv(history_path).set_index("epoch")
        model_directory = history_path.parent / "models"
        available: list[dict[str, object]] = []
        final_paths: list[Path] = []
        for checkpoint in sorted(model_directory.iterdir()):
            match = EPOCH_PATTERN.search(checkpoint.name)
            if match is None:
                final_paths.append(checkpoint)
                continue
            epoch = int(match.group(1))
            if epoch not in history.index:
                continue
            validation_loss = float(history.loc[epoch, "validation_loss"])
            if validation_loss > 0:
                available.append(
                    {
                        "run": history_path.parents[1].name,
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        "checkpoint": str(checkpoint.resolve()),
                        "candidate_reason": "low_validation_loss_shortlist",
                    }
                )
        available.sort(key=lambda row: float(row["validation_loss"]))
        records.extend(available[:candidates_per_run])
        epoch_checkpoints = {str(row["checkpoint"]) for row in available}
        for final_path in final_paths:
            if str(final_path.resolve()) in epoch_checkpoints:
                continue
            records.append(
                {
                    "run": history_path.parents[1].name,
                    "epoch": int(history.index.max()),
                    "validation_loss": float("nan"),
                    "checkpoint": str(final_path.resolve()),
                    "candidate_reason": "final_checkpoint",
                }
            )
    if not records:
        raise ValueError(f"No selectable Cellpose checkpoints found below {run_root}")
    return pd.DataFrame(records).drop_duplicates("checkpoint").reset_index(drop=True)


def main() -> None:
    """Write a checkpoint shortlist whose final ordering will use PQ/F1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-per-run", type=int, default=2)
    args = parser.parse_args()
    result = shortlist_candidates(args.run_root, args.candidates_per_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Shortlisted {len(result)} Cellpose checkpoints for PQ/F1 evaluation")


if __name__ == "__main__":
    main()
