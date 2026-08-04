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

The current dataset is stored in this
[Google Drive folder](https://drive.google.com/drive/folders/15FrdmbZGEWyB2mBgGv2hVpIlkcGQy6tE):

| Drive folder | Content |
|---|---|
| `Astrocytes Training Photos` | ten annotated training images |
| `Astrocytes Training Masks/Astrocytes Final Masks` | matching `*_seg.npy` masks |
| `Astrocytes Morphology Photos` | images reserved for prediction |
| `Astroseg Outputs` | generated QC, annotations, models, and predictions |

Public folder identifiers are documented in `configs/google_drive.yaml`; Google
credentials must never be committed. During a run, files live under the ignored
`.astroseg_runtime/` directory and can be deleted and downloaded again safely.

## Installation

Python `>=3.11,<3.13` is supported.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

The Drive automation also requires [rclone](https://rclone.org/downloads/). After
installing it, create a Google Drive remote once:

```powershell
rclone config
```

Choose `New remote`, name it `astroseg-drive`, choose Google Drive, and complete
the browser sign-in. Do not copy OAuth tokens into this repository.

rclone currently warns that its shared Google client ID is being retired during
2026. The authorized remote works now, but configure a private Google client ID
in rclone before that shared client is disabled.

## Run the complete Drive workflow

From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_drive_pipeline.ps1 `
  -Annotator "your-name"
```

That single command performs the operational workflow:

1. downloads training images, masks, and test images from Drive;
2. creates a manifest with explicit `train` and `test` rows;
3. selects GFAP/DAPI channels and detects nuclei automatically;
4. validates and imports the training instance masks without changing originals;
5. trains fold 0 with grouped cross-validation;
6. predicts individual astrocytes on every test image;
7. compresses the complete `.astroseg_runtime/outputs` tree and uploads a dated
   `.tar.zst` archive to `Astroseg Outputs`.

Use `-Fold 1` (through `-Fold 4`) to train another validation fold. Use
`-SkipDownload` when the input files are already synchronized, or `-SkipUpload`
when testing locally. Training can take a long time; a CUDA-capable GPU is strongly
recommended.

The synchronization operations can also be run separately:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_google_drive.ps1 -Action Download
powershell -ExecutionPolicy Bypass -File scripts/sync_google_drive.ps1 -Action Upload
powershell -ExecutionPolicy Bypass -File scripts/sync_google_drive.ps1 -Action UploadExpanded
```

`Upload` creates a compact dated backup archive. `UploadExpanded` places every
generated file into its matching `metadata/`, `interim/`, `annotations/`,
`checkpoints/`, or `instance_predictions/` Drive subfolder.

## Output layout

```text
.astroseg_runtime/
    dataset/                         downloaded inputs; never modified
        training_images/
        training_masks/
        test_images/
    outputs/                         uploaded to Drive after a run
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
Predictions never overwrite reviewed masks.

The upload command archives the full tree because the uncompressed intermediate
arrays are several gigabytes and contain hundreds of small files. Extract an
archive with `tar --zstd -xf <archive-name>.tar.zst`; its top-level directory is
`outputs/`. The local temporary archive is removed only after a successful upload.

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

The Drive workflow uses `configs/train_instances_drive.yaml`. The standard local
template remains in `configs/train_instances.yaml`.

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

## Repository map

```text
configs/                 local and Drive-backed pipeline settings
src/astroseg/io/         BMP/TIFF/OME-TIFF and manifest loading
src/astroseg/preprocessing/ channels, nuclei, patches, instance targets
src/astroseg/datasets/   aligned semantic and instance datasets
src/astroseg/models/     binary and nucleus-guided U-Nets
src/astroseg/training/   losses, metrics, grouped folds, trainers
src/astroseg/inference/  patch and full-image prediction
src/astroseg/postprocessing/ complete-cell reconstruction
src/astroseg/analysis/   per-cell morphology measurements
scripts/                 command-line and Drive workflows
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

- No validated trained checkpoint is committed; it must be learned from the Drive
  annotations.
- DAPI nucleus detection is automatic but classical and should be checked in QC
  when staining, magnification, or acquisition changes.
- Process crossings may be inherently ambiguous in 2D; some cases require a
  Z-stack or additional markers.
- Measurements are only as reliable as the instance and compartment masks.

Raw images, Cellpose exports, and original human annotations are never overwritten
by preparation, training, or prediction.
