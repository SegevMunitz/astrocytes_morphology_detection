"""Regression tests for object-metric checkpoint selection."""

import json
from pathlib import Path

import pandas as pd

from scripts.rank_instance_checkpoint_evaluations import rank_evaluations
from scripts.select_cellpose_checkpoint_candidates import shortlist_candidates


def _write_evaluation(root: Path, name: str, pq: float, f1: float) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "evaluation.json").write_text(
        json.dumps({"run_directory": "run", "checkpoint_name": f"{name}.pt"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "image_id": "held_out",
                "iou_threshold": 0.5,
                "threshold_selection": "tuning_subset",
                "f1": f1,
                "panoptic_quality": pq,
                "precision": 0.8,
                "recall": 0.9,
            }
        ]
    ).to_csv(directory / "metrics_by_image.csv", index=False)


def test_checkpoint_ranking_prioritizes_panoptic_quality_over_pixel_loss(
    tmp_path: Path,
) -> None:
    _write_evaluation(tmp_path, "low_pq", 0.5, 0.95)
    _write_evaluation(tmp_path, "high_pq", 0.7, 0.8)

    ranking = rank_evaluations(tmp_path, tmp_path / "selection")

    assert ranking.iloc[0]["checkpoint_name"] == "high_pq.pt"
    best = json.loads((tmp_path / "selection" / "best_checkpoint.json").read_text())
    assert best["checkpoint_name"] == "high_pq.pt"


def test_cellpose_shortlist_ignores_zero_placeholder_validation_losses(
    tmp_path: Path,
) -> None:
    result = tmp_path / "lr_0p001" / "result"
    models = result / "models"
    models.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": [25, 50, 100, 299],
            "train_loss": [0.5, 0.4, 0.3, 0.2],
            "validation_loss": [0.0, 0.55, 0.50, 0.0],
        }
    ).to_csv(result / "history.csv", index=False)
    for name in (
        "cellpose_3ch_epoch_0025",
        "cellpose_3ch_epoch_0050",
        "cellpose_3ch_epoch_0100",
        "cellpose_3ch",
    ):
        (models / name).touch()

    candidates = shortlist_candidates(tmp_path, candidates_per_run=1)

    assert candidates["checkpoint"].str.endswith("epoch_0100").any()
    assert not candidates["checkpoint"].str.endswith("epoch_0025").any()
    assert (candidates["candidate_reason"] == "final_checkpoint").any()
