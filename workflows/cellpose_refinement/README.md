# Cellpose/Cyto2 refinement

This workflow starts from a Cellpose checkpoint, expands the input layer from
GFAP/DAPI to GFAP/GFP/DAPI, and fine-tunes every network weight. It does not use
the independent AstroSeg architecture, losses, trainer, or reconstruction code.

The default source is the selected refined Cyto2 checkpoint. Override it with
`CELLPOSE_TRANSFER_SOURCE` when testing another parent checkpoint.

From the repository root on the cluster:

```bash
export ASTROSEG_DATA_ROOT="$HOME/astroseg_data"
export CELLPOSE_TRANSFER_SOURCE="$ASTROSEG_DATA_ROOT/outputs/cellpose/lr_sweeps/lr_sweep_original_channels_20260810/cyto2_cp3/lr_0p25/result/models/cyto2_cp3_lr_0p25_epoch_0450"
export CELLPOSE_TRANSFER_RUN_NAME="three_channel_transfer_$(date +%Y%m%d)"
bash scripts/submit_training_workflows.sh refine-cellpose
```

After training, shortlist checkpoints, evaluate them with held-out masks, rank by
panoptic quality and F1, and then predict test images using the commands in the
root `README.md`.

Both model families can train concurrently:

```bash
bash scripts/submit_training_workflows.sh both
```

The two jobs write to separate output trees and use separate Python environments.
