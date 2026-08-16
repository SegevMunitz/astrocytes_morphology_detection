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

Every model input has three aligned fluorescence planes in one fixed biological
order: normalized GFAP/Cy5, GFP/auxiliary fluorescence, and DAPI. Acquisitions
without GFP receive an explicit all-zero middle plane; transmitted-light planes
are excluded. A residual GroupNorm U-Net then predicts:

- semantic compartments (background, nucleus, soma, process);
- boundaries between touching cells;
- a two-dimensional ownership vector from each cell pixel toward its nucleus.

The ownership prediction is what allows a process to remain assigned to its cell
even when it passes near another nucleus. The older binary U-Net is retained as a
foreground baseline, but it cannot separate individual cells.

```text
image -> metadata-aware GFAP/GFP/DAPI selection -> three-channel residual U-Net
      -> compartments + boundaries + ownership -> nucleus-aware reconstruction
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

Create one immutable, audited dataset snapshot, then submit grouped
cross-validation. Every fold receives one GPU; `%3` limits the array to three
concurrent GPUs:

```bash
cd ~/astrocytes_morphology_detection
export ASTROSEG_DATA_ROOT="$HOME/astroseg_data"
sbatch scripts/slurm_prepare_multichannel.sh
export ASTROSEG_DATASET_NAME="multichannel_20260816"
export ASTROSEG_RUN_NAME="multichannel_cross_validation"
sbatch --array=0-4%3 scripts/slurm_train_instances.sh
```

After all folds finish, calibrate reconstruction thresholds on the explicit
validation IDs, then predict the reserved test images by averaging all five fold
models:

```bash
export ASTROSEG_TRAIN_RUN_NAME="multichannel_cross_validation"
sbatch scripts/slurm_evaluate_instances.sh
export ASTROSEG_RUN_NAME="multichannel_test_ensemble"
sbatch scripts/slurm_predict_instances.sh
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
        datasets/multichannel_20260816/
            metadata/                manifests, mask audit, and snapshot summary
            interim/                 channels, nuclei, distance maps, QC
            annotations/             preserved imports and derived masks
        checkpoints/                 trained weights and learning history
        evaluations/                 OOF metrics and calibrated thresholds
        instance_predictions/
            raw_heads/               model probabilities and ownership offsets
            labels/                  complete-cell instance IDs
            compartments/            nucleus/soma/process classes
            overlays/                visual QC
            cell_measurements.csv
```

Human annotations and automatic predictions are deliberately stored separately.
Predictions never overwrite reviewed masks. The versioned
`metadata/manifest_instances.csv` contains validated absolute cluster paths, while
the source images and Cellpose exports remain unchanged for provenance.

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
patch extraction. All patches from the same image remain in one fold. The 23
unlabeled training images contribute only through a confidence-filtered EMA
teacher consistency loss; they are never assigned fabricated masks. If several
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

### Cellpose fine-tuning

Cellpose fine-tuning is a separate, simpler workflow for establishing a strong
instance-segmentation baseline. It does not replace the nucleus-guided U-Net above.
Each training image must be beside its Cellpose GUI annotation and share its base
name:

```text
~/astroseg_data/training_images/
    BMP4_24h_20x_20240307_145.tif
    BMP4_24h_20x_20240307_145_seg.npy
```

`prepare_cellpose_training_data.py` checks that each pair exists, has identical
spatial dimensions, contains integer instance IDs, and uses the expected channels.
It creates symlinks for a run and never changes the original image or annotation.
Cellpose trains from the final `masks` array: the GUI's `ismanual` field records
provenance but does not include, exclude, or weight a cell during training.

Create the isolated Cellpose 3 environment once, then submit a single baseline:

```bash
cd ~/astrocytes_morphology_detection
export ASTROSEG_DATA_ROOT="$HOME/astroseg_data"
mkdir -p cluster_logs
sbatch scripts/slurm_setup_cellpose3.sh
sbatch scripts/slurm_train_cellpose3.sh
```

For comparable learning-rate experiments, the committed lists in
`configs/cellpose_split_original_channels/` keep whole images in either training or validation. The
two arrays cover `cyto2_cp3` and `cyto3`; each task uses one A100 GPU and writes its
log, checkpoints, loss history, and run metadata inside its own result folder:

```bash
sbatch scripts/slurm_sweep_cellpose_lr.sh       # LR: 0.02, 0.05, 0.1, 0.2
sbatch scripts/slurm_extend_cellpose_lr.sh      # LR: 0.25 through 0.5

python scripts/summarize_cellpose_sweep.py \
  --sweep-dir "$ASTROSEG_DATA_ROOT/outputs/cellpose/lr_sweeps/lr_sweep_replicate_split"
```

The ranked `summary.csv` reports the best saved checkpoint for each run. Validation
loss is useful for comparing runs made with the same split, but it is not a final
biological-quality score; inspect overlays and report held-out instance metrics
before using a model for scientific conclusions.

### Current experiment

The `multichannel_20260816` snapshot contains 33 training images (10 reviewed
instance masks and 23 unlabeled images), 52 test images, and 3,638 annotated
instances. The current residual model has about 4.28 million trainable parameters
and uses fluorescence augmentation, class-weighted multi-head loss, mixed
precision, learning-rate reduction, early stopping, and EMA teacher consistency.
Scientific comparison is based on held-out instance F1 and panoptic quality, not
training loss. Final scores and test predictions are written under the cluster
`evaluations/` and `instance_predictions/` directories rather than asserted in
documentation before the jobs complete.

## Repository map

```text
configs/                 local and cluster-native pipeline settings
src/astroseg/io/         BMP/TIFF/OME-TIFF and manifest loading
src/astroseg/preprocessing/ channels, nuclei, patches, instance targets
src/astroseg/datasets/   aligned semantic and instance datasets
src/astroseg/models/     binary, nucleus-guided, and multichannel residual U-Nets
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
