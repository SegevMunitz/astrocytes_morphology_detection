"""Generate final three-channel Cellpose instance masks for unlabeled test images."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from cellpose import io, models

from astroseg.models import prepare_three_channel_cellpose_image
from astroseg.visualization import save_instance_overlay


def predict_directory(
    image_directory: Path,
    checkpoint: Path,
    output_directory: Path,
    flow_threshold: float,
    cellprob_threshold: float,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Predict every supported test TIFF and save labels, overlays, and counts."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-resolution Cellpose prediction")
    paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".bmp"}
    )
    if not paths:
        raise ValueError(f"No supported test images found in {image_directory}")
    labels_directory = output_directory / "labels"
    overlays_directory = output_directory / "overlays"
    labels_directory.mkdir(parents=True, exist_ok=True)
    overlays_directory.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(
        gpu=True, pretrained_model=str(checkpoint), nchan=3
    )
    records = []
    for path in paths:
        image = prepare_three_channel_cellpose_image(io.imread(str(path)))
        label_path = labels_directory / f"{path.stem}.tiff"
        overlay_path = overlays_directory / f"{path.stem}.png"
        if not overwrite and (label_path.exists() or overlay_path.exists()):
            raise FileExistsError(f"Refusing to overwrite prediction for {path.stem!r}")
        masks, _, _ = model.eval(
            image,
            channels=None,
            channel_axis=0,
            diameter=None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        labels = np.asarray(masks, dtype=np.uint32)
        tifffile.imwrite(label_path, labels, photometric="minisblack")
        save_instance_overlay(image[0], labels, overlay_path)
        records.append(
            {
                "image_id": path.stem,
                "source_path": str(path.resolve()),
                "prediction_path": str(label_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "cell_count": int(np.unique(labels[labels > 0]).size),
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "checkpoint": str(checkpoint.resolve()),
            }
        )
    report = pd.DataFrame(records)
    report.to_csv(output_directory / "prediction_report.csv", index=False)
    return report


def main() -> None:
    """Parse final checkpoint, tuned thresholds, and test output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = predict_directory(
        args.image_dir,
        args.checkpoint,
        args.output_dir,
        args.flow_threshold,
        args.cellprob_threshold,
        args.overwrite,
    )
    print(f"Saved {len(report)} test predictions containing {report.cell_count.sum()} cells")


if __name__ == "__main__":
    main()
