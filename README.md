# Astrocyte Instance Segmentation

This project separates complete individual GFAP-positive astrocytes in BMP, TIFF,
and OME-TIFF microscopy images. Its goal is not only to find astrocyte signal, but
to associate each nucleus, soma, and process with the correct cell.

The final prediction contains:

```text
instance labels:  0 = background, 1..N = individual astrocytes
compartments:     0 = background, 1 = nucleus, 2 = soma, 3 = process
ownership:        every predicted cell is linked to one nucleus
```

## Model overview

Every model input has three aligned channels: normalized GFAP intensity, a binary
nucleus mask, and a nucleus-proximity map. A shared U-Net then predicts:

- semantic compartments (background, nucleus, soma, process);
- boundaries between touching cells;
- a two-dimensional ownership vector from each cell pixel toward its nucleus.

The ownership prediction is what allows a process to remain assigned to its cell
even when it passes near another nucleus. The older binary U-Net is retained as a
foreground baseline, but it cannot separate individual cells.

```text
image -> automatic GFAP/DAPI selection -> nucleus detection
      -> nucleus-guided U-Net -> compartments + boundaries + ownership
      -> complete individual astrocyte instances
```

## Data storage

Git contains only code, configuration, tests, and notebooks. Large microscopy
images, masks, intermediate arrays, checkpoints, and predictions are intentionally
excluded by `.gitignore`.

The operational dataset now lives on the HUJI ELSC filesystem under the account
that submits the Slurm jobs:

```text
~/astroseg_data/
    training_images/       original annotated microscopy images
    training_masks/        original Cellpose/manual *_seg.npy files
    test_images/           images reserved for prediction
    outputs/               manifests, QC, annotations, checkpoints, predictions
```

Set `ASTROSEG_DATA_ROOT` when using another location. Cluster YAML files expand
that variable at runtime, so neither the repository nor its manifests require a
Google Drive mount, rclone, or embedded account path. The original Drive folder
is retained as a migration source/backup only; cluster execution does not access it.

## Installation

Python `>=3.11,<3.13` is supported.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

## Run on the HUJI cluster

The cluster copy of the repository already uses a Python 3.11 environment. For a
new clone, create or update it once through the CPU queue:

```bash
cd ~/astrocytes_morphology_detection
mkdir -p cluster_logs
sbatch scripts/slurm_setup.sh
```

Prepare the migrated manifest once, then submit grouped cross-validation. Every
fold receives one GPU; `%3` limits the array to three concurrent GPUs:

```bash
cd ~/astrocytes_morphology_detection
export ASTROSEG_DATA_ROOT="$HOME/astroseg_data"
bash scripts/prepare_cluster_storage.sh
sbatch --array=0-4%3 scripts/slurm_train_instances.sh
```

After selecting a checkpoint, predict all reserved test images with one GPU:

```bash
export ASTROSEG_CHECKPOINT="$ASTROSEG_DATA_ROOT/outputs/checkpoints/cross_validation/fold_1/best.pt"
export ASTROSEG_RUN_NAME="review_round_1"
sbatch --export=ALL,ASTROSEG_DATA_ROOT,ASTROSEG_CHECKPOINT,ASTROSEG_RUN_NAME \
  scripts/slurm_predict_instances.sh
```

All artifacts remain under `~/astroseg_data/outputs`. Human annotations and
automatic predictions are always separate, and no command uploads to Drive.

## Output layout

```text
~/astroseg_data/
    training_images/                 original inputs; never modified
    training_masks/                  preserved source masks
    test_images/                     prediction-only images
    outputs/
        metadata/                    manifests and split assignments
        interim/                     channels, nuclei, distance maps, QC
        annotations/                 preserved imports and derived masks
        checkpoints/                 trained weights and learning history
        instance_predictions/
            raw_heads/               model probabilities and ownership offsets
            labels/                  complete-cell instance IDs
            compartments/            nucleus/soma/process classes
            overlays/                visual QC
            cell_measurements.csv
```

Human annotations and automatic predictions are deliberately stored separately.
Predictions never overwrite reviewed masks. `manifest_instances_cluster.csv`
contains validated absolute cluster paths; the original imported manifest remains
unchanged for provenance.

## Annotation contract

The supplied `*_seg.npy` files are matched by basename. For example,
`BMP4_24h_20x_20240307_145_seg.npy` belongs to image ID
`BMP4_24h_20x_20240307_145`.

A trainable complete-cell mask must be:

- a two-dimensional integer array with exactly the image height and width;
- `0` for background and one positive ID per astrocyte;
- one consistent ID across that cell's nucleus, soma, and processes;
- aligned with the image; a unique detected nucleus should lie near each cell soma.

The importer preserves the original Cellpose file, derives a compatibility binary
mask, generates a QC overlay, and records provenance in the manifest. Because GFAP
often surrounds rather than overlaps the nucleus, cells and nuclei are paired by
unique nearby centroids. Unmatched fragments still supervise segmentation but are
excluded from ownership-vector loss. A mask that contains only nuclei is not enough
to train complete-cell process ownership.

Annotation states are `none`, `seed`, `pseudo`, `corrected`, and `reviewed`. Only
`seed`, `corrected`, and `reviewed` enter training by default. Important manifest
fields are:

| Field | Meaning |
|---|---|
| `annotation_path` | derived binary GFAP mask |
| `instance_annotation_path` | human complete-cell instance IDs |
| `annotation_status` | annotation lifecycle state |
| `annotation_source`, `annotator` | provenance |
| `review_status` | review state |
| `split` | `train`, `val`, or `test` |

## Training and validation

Slurm uses `configs/train_instances_cluster.yaml`, whose storage paths come from
`ASTROSEG_DATA_ROOT`. The standard local template remains in
`configs/train_instances.yaml`.

Because there are only ten labeled images, cross-validation is grouped before
patch extraction. All patches from the same image remain in one fold. If several
images later come from the same biological well, add `well_id` to the manifest and
change `group_column` from `image_id` to `well_id`.

Run a quick engineering check without real data:

```powershell
.\.python311\python.exe scripts/train_instances.py --smoke-test
```

The intended annotation loop is:

```text
seed masks -> initial training -> predictions on unlabeled images
           -> manual correction -> corrected/reviewed import -> retraining
```

### Current baseline status

The present nucleus-guided U-Net has 1.93 million trainable parameters. Grouped
five-fold validation on the ten annotated images produced mean semantic Dice
`0.813` and mean boundary Dice `0.203`. It is therefore a useful annotation
bootstrap model, but cell separation and process ownership are not yet validated
well enough for final scientific measurements.

The architecture is intentionally unchanged during the storage migration. A
larger U-Net or transformer would add capacity while the supervision set remains
only ten images, increasing overfitting risk without addressing inconsistent or
ambiguous process ownership. Reconsider architecture complexity after adding
diverse corrected instances and measuring held-out instance F1/panoptic quality.

## Repository map

```text
configs/                 local and cluster-native pipeline settings
src/astroseg/io/         BMP/TIFF/OME-TIFF and manifest loading
src/astroseg/preprocessing/ channels, nuclei, patches, instance targets
src/astroseg/datasets/   aligned semantic and instance datasets
src/astroseg/models/     binary and nucleus-guided U-Nets
src/astroseg/training/   losses, metrics, grouped folds, trainers
src/astroseg/inference/  patch and full-image prediction
src/astroseg/postprocessing/ complete-cell reconstruction
src/astroseg/analysis/   per-cell morphology measurements
scripts/                 command-line and Slurm workflows
notebooks/               guided execution and learning
tests/                   synthetic regression tests
```

Reusable logic lives under `src/astroseg/`; scripts only connect that logic to
configuration and file paths.

## Tests

```powershell
.\.python311\python.exe -m pytest -q
```

Tests cover mixed BMP/TIFF loading (including incorrect filename extensions),
channel and nucleus preparation, annotation preservation, grouped folds, instance
targets, model heads, losses, ownership offsets, object metrics, and notebooks.

## Current limitations

- Checkpoints are stored on the cluster rather than committed to Git. The current
  five-fold model is a baseline trained from only ten annotated images; it is not
  yet sufficient for final biological conclusions.
- DAPI nucleus detection is automatic but classical and should be checked in QC
  when staining, magnification, or acquisition changes.
- Process crossings may be inherently ambiguous in 2D; some cases require a
  Z-stack or additional markers.
- Measurements are only as reliable as the instance and compartment masks.

Raw images, Cellpose exports, and original human annotations are never overwritten
by preparation, training, or prediction.
