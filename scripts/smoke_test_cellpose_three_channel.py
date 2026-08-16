"""Verify real Cellpose checkpoint expansion before submitting GPU training."""

import argparse
from pathlib import Path

import torch

from train_cellpose_model import _build_three_channel_model


def main() -> None:
    """Load one exact checkpoint and validate its expanded input tensor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    model, expanded_keys = _build_three_channel_model(str(args.checkpoint))
    for expanded_key in expanded_keys:
        weight = model.net.state_dict()[expanded_key]
        if model.net.nchan != 3 or weight.shape[1] != 3:
            raise RuntimeError("Expanded Cellpose model does not have three input channels")
        if torch.count_nonzero(weight[:, 1]).item() != 0:
            raise RuntimeError("GFP input weights must be exactly zero at initialization")
        if torch.count_nonzero(weight[:, (0, 2)]).item() == 0:
            raise RuntimeError("Transferred GFAP/DAPI weights are unexpectedly empty")
    print(f"Three-channel Cellpose transfer smoke passed: {expanded_keys}")


if __name__ == "__main__":
    main()
