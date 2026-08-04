"""Structural regression tests for the executable notebook guides."""

import json
from pathlib import Path

import pytest


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED_NOTEBOOKS = (
    "00_run_pipeline.ipynb",
    "01_inspect_tiff_channels.ipynb",
    "02_visualize_nucleus_masks.ipynb",
    "03_create_initial_annotations.ipynb",
    "04_evaluate_baseline.ipynb",
)


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_guided_notebook_is_populated_and_code_compiles(filename: str) -> None:
    """Every documented notebook contains prose and valid executable Python.

    Compiling cells without running research operations catches malformed JSON and
    syntax regressions while keeping the test suite independent of microscopy data.
    """
    path = NOTEBOOK_DIRECTORY / filename
    with path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    assert notebook["nbformat"] == 4
    markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert markdown_cells
    assert code_cells
    for index, cell in enumerate(code_cells, start=1):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{filename}:code-cell-{index}", "exec")


def test_notebook_sequence_uses_current_nucleus_terminology() -> None:
    """The obsolete Cellpose-specific placeholder is not part of the guide."""
    assert not (NOTEBOOK_DIRECTORY / "02_visualize_cellpose_masks.ipynb").exists()
