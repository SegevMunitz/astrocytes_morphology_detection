"""Generate initial GFAP pseudo labels without a trained checkpoint."""

import argparse
from pathlib import Path

import pandas as pd

from astroseg.annotations import save_pseudo_label_artifacts
from astroseg.io import get_channel, load_manifest, load_ome_tiff, validate_manifest
from astroseg.preprocessing import detect_gfap_bootstrap_mask


def _resolve_path(value: str, manifest_path: Path) -> Path:
    """Resolve a microscopy path using project- and manifest-relative layouts.

    The original manifest value is retained in the error so a failed batch row is
    easy to locate and correct.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Microscopy image does not exist: {value}")


def _portable_path(path: Path) -> str:
    """Store an output path relative to the project whenever possible.

    Forward slashes make generated manifests portable across supported platforms,
    while files outside the project retain explicit paths.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def generate_bootstrap_pseudo_labels(
    manifest_path: Path,
    output_directory: Path,
    output_manifest: Path,
    split: str | None = None,
    overwrite: bool = False,
    gaussian_sigma: float = 0.8,
    low_threshold_ratio: float = 0.4,
    high_threshold_scale: float = 1.0,
    min_component_area: int = 24,
    max_hole_area: int = 24,
) -> pd.DataFrame:
    """Create heuristic proposals for every selected unannotated image.

    Automatic artifacts stay under the output directory, and a separate manifest
    marks them ``pseudo`` so they cannot enter supervised training by default.
    """
    manifest = load_manifest(manifest_path)
    selected = manifest["annotation_status"] == "none"
    if split is not None:
        selected &= manifest["split"] == split
    indices = manifest.index[selected].tolist()
    if not indices:
        raise ValueError("No unannotated manifest rows match the requested split")

    records: list[dict[str, object]] = []
    for index in indices:
        row = manifest.loc[index]
        image_id = str(row["image_id"])
        gfap_channel = str(row["gfap_channel"]).strip()
        if not gfap_channel:
            raise ValueError(
                f"GFAP channel is empty for {image_id!r}; run prepare_dataset.py first"
            )
        microscopy = load_ome_tiff(
            _resolve_path(str(row["path"]), manifest_path)
        )
        gfap = get_channel(microscopy, gfap_channel)
        result = detect_gfap_bootstrap_mask(
            gfap,
            gaussian_sigma=gaussian_sigma,
            low_threshold_ratio=low_threshold_ratio,
            high_threshold_scale=high_threshold_scale,
            min_component_area=min_component_area,
            max_hole_area=max_hole_area,
        )
        artifacts = save_pseudo_label_artifacts(
            image_id,
            result.probabilities,
            gfap,
            output_directory,
            overwrite=overwrite,
            hard_mask=result.mask,
        )
        manifest.loc[index, "annotation_path"] = _portable_path(artifacts.mask_path)
        manifest.loc[index, "annotation_status"] = "pseudo"
        manifest.loc[index, "annotation_source"] = "heuristic:gfap_hysteresis_otsu"
        manifest.loc[index, "annotator"] = ""
        manifest.loc[index, "review_status"] = "pending"
        records.append(
            {
                "image_id": image_id,
                "low_threshold": result.low_threshold,
                "high_threshold": result.high_threshold,
                "foreground_fraction": result.foreground_fraction,
                "gaussian_sigma": gaussian_sigma,
                "low_threshold_ratio": low_threshold_ratio,
                "high_threshold_scale": high_threshold_scale,
                "min_component_area": min_component_area,
                "max_hole_area": max_hole_area,
                "mask_path": _portable_path(artifacts.mask_path),
                "overlay_path": _portable_path(artifacts.overlay_path),
            }
        )

    validate_manifest(manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    pd.DataFrame(records).to_csv(output_directory / "bootstrap_report.csv", index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse batch paths and configurable heuristic segmentation parameters.

    Defaults operate on the standard project layout and keep the generated
    manifest beside the automatic artifacts rather than replacing source metadata.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/metadata/manifest.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/pseudo_labels")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("outputs/pseudo_labels/manifest.csv"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gaussian-sigma", type=float, default=0.8)
    parser.add_argument("--low-threshold-ratio", type=float, default=0.4)
    parser.add_argument("--high-threshold-scale", type=float, default=1.0)
    parser.add_argument("--min-component-area", type=int, default=24)
    parser.add_argument("--max-hole-area", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    """Generate proposals and report the separate pseudo-manifest location.

    The command never changes the source manifest or any human-annotation file.
    Existing automatic artifacts require an explicit overwrite flag.
    """
    args = parse_args()
    manifest = generate_bootstrap_pseudo_labels(
        args.manifest,
        args.output_dir,
        args.output_manifest,
        args.split,
        args.overwrite,
        args.gaussian_sigma,
        args.low_threshold_ratio,
        args.high_threshold_scale,
        args.min_component_area,
        args.max_hole_area,
    )
    count = int((manifest["annotation_status"] == "pseudo").sum())
    print(f"Recorded {count} bootstrap pseudo labels in {args.output_manifest}")


if __name__ == "__main__":
    main()
