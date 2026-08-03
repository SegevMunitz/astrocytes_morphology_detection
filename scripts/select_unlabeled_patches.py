"""Rank unreviewed patches for manual correction using predictive entropy."""

import argparse
from pathlib import Path

from astroseg.annotations import select_uncertain_patches
from astroseg.io import load_manifest


def parse_args() -> argparse.Namespace:
    """Parse uncertainty-selection inputs, patch geometry, and ranking limits.

    Multiple annotation states may be selected, and an optional per-image cap
    prevents one uncertain field from consuming the entire correction queue.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--statuses", nargs="+", default=["none", "pseudo"])
    parser.add_argument("--max-patches-per-image", type=int)
    return parser.parse_args()


def main() -> None:
    """Rank eligible patches and write the resulting manual-correction queue.

    Selection uses saved probability tensors rather than hard masks, and the
    printed count reflects the final globally ranked CSV rows.
    """
    args = parse_args()
    selection = select_uncertain_patches(
        load_manifest(args.manifest),
        args.probability_dir,
        args.patch_size,
        args.overlap,
        args.top_k,
        args.statuses,
        args.max_patches_per_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output, index=False)
    print(f"Wrote {len(selection)} ranked patches to {args.output}")


if __name__ == "__main__":
    main()
