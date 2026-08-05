"""Summarize completed Cellpose sweep histories and selectable checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize_sweep(sweep_directory: Path) -> list[dict[str, str | int | float]]:
    """Return validation minima for every run and verify saved checkpoint paths."""
    summaries: list[dict[str, str | int | float]] = []
    for metadata_path in sorted(sweep_directory.glob("*/lr_*/result/run.json")):
        result_directory = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with (result_directory / "history.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        evaluated = [row for row in rows if float(row["validation_loss"]) > 0]
        if not evaluated:
            raise ValueError(f"No validation losses found in {result_directory}")
        best = min(evaluated, key=lambda row: float(row["validation_loss"]))
        save_every = int(metadata["save_every"])
        saved_rows = [
            row
            for row in evaluated
            if int(row["epoch"]) > 0 and int(row["epoch"]) % save_every == 0
        ]
        model_name = Path(str(metadata["model_path"])).name
        existing_saved = [
            row
            for row in saved_rows
            if (result_directory / "models" / f"{model_name}_epoch_{int(row['epoch']):04d}").is_file()
        ]
        if not existing_saved:
            raise ValueError(f"No selectable checkpoints found in {result_directory}")
        checkpoint = min(existing_saved, key=lambda row: float(row["validation_loss"]))
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_path = (
            result_directory / "models" / f"{model_name}_epoch_{checkpoint_epoch:04d}"
        )
        summaries.append(
            {
                "pretrained_model": str(metadata["pretrained_model"]),
                "learning_rate": float(metadata["learning_rate"]),
                "best_logged_epoch": int(best["epoch"]),
                "best_logged_validation_loss": float(best["validation_loss"]),
                "best_checkpoint_epoch": checkpoint_epoch,
                "best_checkpoint_validation_loss": float(checkpoint["validation_loss"]),
                "best_checkpoint_path": str(checkpoint_path),
                "final_model_path": str(metadata["model_path"]),
            }
        )
    return sorted(
        summaries,
        key=lambda row: float(row["best_checkpoint_validation_loss"]),
    )


def parse_args() -> argparse.Namespace:
    """Parse the sweep root and optional summary output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Write a ranked CSV summary and print the same ranking to stdout."""
    args = parse_args()
    summaries = summarize_sweep(args.sweep_dir)
    if not summaries:
        raise ValueError(f"No completed Cellpose runs found in {args.sweep_dir}")
    output = args.output or args.sweep_dir / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(
            f"{row['pretrained_model']} lr={row['learning_rate']}: "
            f"checkpoint epoch {row['best_checkpoint_epoch']}, "
            f"validation_loss={row['best_checkpoint_validation_loss']:.6f}"
        )
    print(f"Summary written to {output}")


if __name__ == "__main__":
    main()
