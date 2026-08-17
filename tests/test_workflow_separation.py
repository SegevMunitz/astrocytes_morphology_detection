"""Regression tests for the independent and Cellpose workflow boundary."""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_isolated_import(module: str, forbidden: str) -> None:
    code = (
        f"import {module}; import sys; "
        f"assert not any(name == '{forbidden}' or name.startswith('{forbidden}.') "
        "for name in sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True)


def test_cellpose_import_does_not_load_independent_models_or_trainers() -> None:
    _assert_isolated_import("astroseg.cellpose", "astroseg.models")
    _assert_isolated_import("astroseg.cellpose", "astroseg.training")


def test_independent_models_do_not_load_cellpose_workflow() -> None:
    _assert_isolated_import("astroseg.models", "astroseg.cellpose")


def test_cluster_launcher_exposes_separate_and_concurrent_workflows() -> None:
    launcher = (PROJECT_ROOT / "scripts" / "submit_training_workflows.sh").read_text(
        encoding="utf-8"
    )
    assert "new-model)" in launcher
    assert "refine-cellpose)" in launcher
    assert "both)" in launcher
    assert "slurm_sweep_astroseg_v2.sh" in launcher
    assert "slurm_train_cellpose_three_channel.sh" in launcher
