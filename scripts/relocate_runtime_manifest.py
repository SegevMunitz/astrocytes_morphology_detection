"""Relocate a synchronized runtime manifest into canonical cluster storage."""

import argparse
from pathlib import Path, PurePosixPath

import pandas as pd

from astroseg.io import load_manifest, validate_manifest


_RUNTIME_PREFIXES = {
    ".astroseg_runtime/dataset/training_images": "training_images",
    ".astroseg_runtime/dataset/training_masks": "training_masks",
    ".astroseg_runtime/dataset/test_images": "test_images",
    ".astroseg_runtime/outputs": "outputs",
}


def _relocate_path(value: str, data_root: Path) -> str:
    """Map one legacy runtime path into a canonical cluster data directory.

    Absolute and unrelated relative paths remain unchanged. Path separators are
    normalized first so manifests created on Windows relocate safely on Linux.
    """
    text = str(value).strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    for prefix, destination in _RUNTIME_PREFIXES.items():
        if normalized == prefix or normalized.startswith(prefix + "/"):
            suffix = PurePosixPath(normalized).relative_to(prefix)
            return (data_root / destination / Path(*suffix.parts)).resolve().as_posix()
    return text


def relocate_runtime_manifest(
    input_path: Path,
    output_path: Path,
    data_root: Path,
    validate_files: bool = True,
) -> pd.DataFrame:
    """Rewrite every manifest path column and optionally require files to exist.

    The source manifest is never modified. This preserves provenance while
    creating a cluster-specific manifest beside the migrated output metadata.
    """
    manifest = load_manifest(input_path)
    path_columns = [
        column for column in manifest.columns if column == "path" or column.endswith("_path")
    ]
    relocated = manifest.copy()
    root = data_root.expanduser().resolve()
    for column in path_columns:
        relocated[column] = relocated[column].map(lambda value: _relocate_path(value, root))
    validate_manifest(relocated)
    if validate_files:
        missing: list[str] = []
        for _, row in relocated.iterrows():
            for column in path_columns:
                value = str(row[column]).strip()
                if value and not Path(value).is_file():
                    missing.append(f"{row['image_id']}:{column}={value}")
        if missing:
            preview = "; ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"Relocated manifest references {len(missing)} missing files: {preview}{suffix}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relocated.to_csv(output_path, index=False)
    return relocated


def parse_args() -> argparse.Namespace:
    """Parse source manifest, canonical data root, and destination options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--skip-file-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Write a validated cluster manifest without changing its source file."""
    args = parse_args()
    frame = relocate_runtime_manifest(
        args.input,
        args.output,
        args.data_root,
        validate_files=not args.skip_file_validation,
    )
    print(f"Relocated {len(frame)} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
