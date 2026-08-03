"""Extract preliminary field-level morphology features from predicted class masks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from astroseg.analysis import extract_image_features


def extract_feature_table(mask_directory: Path, output_path: Path, positive_class: int = 1) -> pd.DataFrame:
    """Measure positive area, components, and skeleton topology for TIFF masks."""
    if not mask_directory.is_dir():
        raise NotADirectoryError(f"Mask directory does not exist: {mask_directory}")
    paths = sorted(
        path for path in mask_directory.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not paths:
        raise FileNotFoundError(f"No TIFF masks found in {mask_directory}")
    records = []
    for path in paths:
        mask = tifffile.imread(path)
        features = extract_image_features(np.asarray(mask) == positive_class)
        records.append({"image_id": path.stem, **features})
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-class", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Write the preliminary field-level feature table."""
    args = parse_args()
    frame = extract_feature_table(args.mask_dir, args.output, args.positive_class)
    print(f"Wrote preliminary field-level features for {len(frame)} images to {args.output}")


if __name__ == "__main__":
    main()
