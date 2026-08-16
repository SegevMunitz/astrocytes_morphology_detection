"""Headless tests for the desktop mask-review data and editing layer."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from astroseg.gui import (
    MaskReviewDataset,
    load_instance_mask,
    paint_instance_disk,
    render_instance_overlay,
    save_corrected_instances,
)


def _write_image(path: Path, value: int = 1) -> None:
    tifffile.imwrite(path, np.full((2, 9, 11), value, dtype=np.uint16))


def _write_mask(path: Path, label: int = 1) -> None:
    mask = np.zeros((9, 11), dtype=np.uint16)
    mask[2:6, 3:8] = label
    tifffile.imwrite(path, mask)


def test_dataset_discovers_exact_images_models_and_cellpose_truth(tmp_path: Path) -> None:
    images = tmp_path / "images"
    first_model = tmp_path / "first"
    second_model = tmp_path / "second"
    truth = tmp_path / "truth"
    for directory in (images, first_model, second_model, truth):
        directory.mkdir()

    _write_image(images / "field_a.tif")
    _write_image(images / "field_b.tiff")
    _write_mask(first_model / "field_a.tiff")
    _write_mask(first_model / "field_b.tiff")
    _write_mask(second_model / "field_a.tif", label=4)
    payload = {"masks": load_instance_mask(first_model / "field_a.tiff")}
    np.save(truth / "field_a_seg.npy", payload, allow_pickle=True)

    dataset = MaskReviewDataset.discover(
        images,
        {"First": first_model, "Second": second_model},
        truth,
    )

    assert dataset.image_ids == ["field_a", "field_b"]
    assert dataset.model_names == ["First", "Second"]
    assert set(dataset.model_masks["Second"]) == {"field_a"}
    assert set(dataset.ground_truth) == {"field_a"}


def test_render_overlay_and_selected_outline() -> None:
    image = np.arange(99, dtype=np.float32).reshape(9, 11)
    labels = np.zeros((9, 11), dtype=np.uint16)
    labels[2:7, 3:8] = 7

    rendered = render_instance_overlay(
        image, labels, alpha=0.7, show_outlines=True, selected_label=7
    )

    assert rendered.shape == (9, 11, 3)
    assert rendered.dtype == np.uint8
    assert np.any(rendered[..., 0] != rendered[..., 1])
    assert np.any(np.all(rendered == np.array([255, 230, 0]), axis=-1))
    with pytest.raises(ValueError, match="aligned"):
        render_instance_overlay(image, labels[:-1])


def test_paint_and_erase_clipped_disks() -> None:
    labels = np.zeros((9, 11), dtype=np.uint32)
    paint_instance_disk(labels, y=0, x=0, radius=3, label_id=12)
    assert labels[0, 0] == 12
    assert labels[-1, -1] == 0

    paint_instance_disk(labels, y=0, x=0, radius=1, label_id=0)
    assert labels[0, 0] == 0
    assert np.any(labels == 12)


def test_save_correction_relabels_and_writes_cellpose_export(tmp_path: Path) -> None:
    labels = np.zeros((9, 11), dtype=np.uint32)
    labels[1:4, 1:4] = 5
    labels[5:8, 6:10] = 20
    source = tmp_path / "source.tif"
    _write_image(source)
    destination = tmp_path / "corrections"

    result = save_corrected_instances(
        "field_a",
        labels,
        destination,
        source,
        cellpose_channels=(1, 3),
    )

    assert result.cell_count == 2
    assert set(np.unique(load_instance_mask(result.tiff_path))) == {0, 1, 2}
    assert result.cellpose_path is not None
    payload = np.load(result.cellpose_path, allow_pickle=True).item()
    assert payload["chan_choose"] == [1, 3]
    assert payload["ismanual"].tolist() == [True, True]
    assert np.array_equal(payload["masks"], load_instance_mask(result.tiff_path))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_corrected_instances("field_a", labels, destination, source)
