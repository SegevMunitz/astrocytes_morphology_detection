# Astrocyte Instance Segmentation

Pipeline for separating complete individual GFAP-positive astrocytes in BMP, TIFF,
and OME-TIFF microscopy images.

The final output contains:

```text
instance labels:  0 = background, 1..N = individual astrocytes
compartments:     0 = background, 1 = nucleus, 2 = soma, 3 = process
ownership:        each astrocyte ID is linked to one nucleus ID
```

The main challenge is not merely detecting GFAP signal, but assigning every soma
and process to the correct cell. Nuclei are therefore used as cell-identity anchors.

## How the model works

Each model input contains three aligned channels:

| Channel | Content |
|---|---|
| 0 | Normalized GFAP intensity |
| 1 | Binary nucleus mask |
| 2 | Nucleus-proximity map |

`NucleusGuidedInstanceUNet` uses one shared U-Net and three output heads:

```text
microscopy image
    |-- GFAP ---------------------------------------+
    `-- DAPI -> detected nucleus instances          |
                    |-- binary nuclei --------------|
                    `-- proximity map --------------+
                                                     v
                                          shared 2D U-Net
                              +----------------+-----+----------------+
                              |                |                      |
                       compartments       cell boundaries     ownership offsets
                              |                |                      |
                              +----------------+----------------------+
                                               v
                              complete individual astrocyte instances
```

The heads predict:

- semantic compartments: background, nucleus, soma, and process;
- boundaries only where different cell IDs touch;
- a `dy, dx` vector from every cell pixel toward its owning nucleus.

The ownership head is important when a long process passes closer to another
nucleus. Nearest-nucleus assignment alone cannot solve that case.

The previous binary U-Net remains available. It predicts merged GFAP foreground
and is useful for creating an initial proposal, but it does not separate cells.

## Current project status

The complete instance architecture, training, prediction, annotation import,
grouped cross-validation, metrics, and QC code are implemented.

No trained instance checkpoint or human-corrected complete-cell dataset is bundled.
Therefore, the current result for `7d_453` is a nucleus-seeded watershed bootstrap:

```text
outputs/astrocyte_instances/overlays/7d_453_instances.png
outputs/astrocyte_instances/overlays/7d_453_compartments.png
```

It is a proposal for correction, not validated process ownership or a trusted cell
count. Learned ownership begins only after importing complete-cell annotations.

## Repository structure

```text
configs/
    train_instances.yaml            final instance-model configuration
    train_binary.yaml               retained binary foreground baseline

data/
    raw/                             original microscopy files
    interim/                         channels, nucleus labels, proximity maps, QC
    annotations/
        originals/                   preserved human instance masks
        compartment_originals/       preserved optional compartment masks
        binary/                      derived binary masks
        qc/                          annotation overlays
    metadata/                        manifests and annotation pair tables

src/astroseg/
    io/                              TIFF and manifest loading
    preprocessing/                   channels, nuclei, patches, instance targets
    datasets/                        semantic and instance PyTorch datasets
    models/                          binary and nucleus-guided U-Nets
    training/                        losses, trainers, metrics, grouped folds
    inference/                       patch prediction and full-image stitching
    postprocessing/                  instance reconstruction and cleanup
    visualization/                   QC overlays
    analysis/                        per-cell morphology

scripts/                             command-line workflows
notebooks/00_run_pipeline.ipynb      routine operational notebook
tests/                               synthetic regression tests
outputs/                             automatic results and checkpoints
```

Reusable implementation is kept under `src/astroseg/`. Files under `scripts/`
connect configuration and paths to those modules.

## Manifest

The manifest is the central index: one row represents one microscopy image.

Important columns are:

| Column | Meaning |
|---|---|
| `image_id` | Stable image identifier |
| `path` | Original microscopy image |
| `gfap_channel`, `dapi_channel` | Selected image channels |
| `cellpose_mask_path` | Nucleus instance labels; legacy name retained for compatibility |
| `annotation_path` | Binary GFAP mask |
| `instance_annotation_path` | Human complete-cell instance IDs |
| `compartment_annotation_path` | Optional human nucleus/soma/process classes |
| `annotation_status` | `none`, `seed`, `pseudo`, `corrected`, or `reviewed` |
| `annotation_source`, `annotator`, `review_status` | Annotation provenance |
| `split` | `train`, `val`, `test`, or empty |

Only `seed`, `corrected`, and `reviewed` annotations enter training by default.
Automatic `pseudo` predictions are stored separately and never become trusted
targets without human correction.

Extra grouping columns such as `well_id`, `experiment_id`, or
`biological_replicate` may be added.

## Installation

Python `>=3.11,<3.13` is supported.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

Because the package uses a `src/` layout, either install it with `pip install -e .`
or run the repository's configured Python environment.

## Routine automatic run

Place `.bmp`, `.tif`, and `.tiff` images together under `data/raw/`. File format
does not determine the split; `train`, `val`, and `test` are assigned in the
manifest. RGB BMP and ordinary RGB TIFF inputs expose `Red`, `Green`, and `Blue`
channels. OME-TIFF keeps its metadata channel names.

Build the manifest when starting a new dataset:

```powershell
.\.python311\python.exe scripts\build_manifest.py `
  --raw-dir data/raw `
  --output data/metadata/manifest.csv
```

Then run:

```powershell
.\.python311\python.exe scripts\prepare_dataset.py
.\.python311\python.exe scripts\generate_bootstrap_pseudo_labels.py --overwrite
.\.python311\python.exe scripts\generate_astrocyte_instances.py --overwrite
```

These commands automatically:

1. select or extract GFAP and DAPI channels;
2. detect nucleus instances from DAPI;
3. build nucleus masks and proximity maps;
4. generate a binary GFAP proposal;
5. create watershed-based individual-cell proposals and QC overlays.

The same automatic workflow is available in
`notebooks/00_run_pipeline.ipynb`. Review the user-settings cell and choose
**Run All**. The other notebooks are intended for learning and troubleshooting.

## Creating real training annotations

Correct the bootstrap output in an annotation tool. The required instance-mask
contract is:

- a two-dimensional integer TIFF aligned exactly with the original image;
- `0` for background;
- one consistent positive ID across each cell's nucleus, soma, and processes;
- a different ID for every astrocyte;
- every cell ID overlaps exactly one detected nucleus.

Cellpose exports such as
`BMP4_24h_20x_20240307_145_seg.npy` are supported directly. Put them under:

```text
data/manual_exports/
```

The basename before `_seg.npy` must equal the manifest `image_id`. The importer
extracts Cellpose's internal `masks` array and preserves the complete original
file. Because Cellpose uses a pickled NumPy dictionary, import only trusted files.

Import every matching Cellpose export without preparing a CSV:

```powershell
.\.python311\python.exe scripts\import_instance_annotations.py `
  --manifest data/metadata/manifest.csv `
  --cellpose-dir data/manual_exports `
  --output-manifest data/metadata/manifest_instances.csv `
  --annotator Segev
```

For the example filename, the manifest must contain the image ID
`BMP4_24h_20x_20240307_145`.

An optional compartment mask may use:

```text
0 = background
1 = nucleus
2 = soma
3 = process
```

If filenames do not match the image IDs, or if compartment masks are also
available, create a pair table instead:

```csv
image_id,instance_mask_path,compartment_mask_path,annotation_status,annotator,review_status
7d_453,exports/7d_453_cells.tiff,exports/7d_453_compartments.tiff,seed,AB,pending
```

Import without modifying the original export:

```powershell
.\.python311\python.exe scripts\import_instance_annotations.py `
  --manifest data/metadata/manifest.csv `
  --pairs-csv data/metadata/instance_pairs.csv `
  --output-manifest data/metadata/manifest_instances.csv `
  --annotator AB
```

The importer validates dimensions and cell-to-nucleus mapping, preserves the
original files, derives a binary mask for compatibility, and produces QC overlays.

## Training

Update `configs/train_instances.yaml`:

```yaml
data:
  manifest_path: data/metadata/manifest_instances.csv

cross_validation:
  enabled: true
  n_splits: 5
  group_column: image_id   # use well_id when several images share a well
```

All patches from one image or well inherit the same fold. Fold assignment happens
before patch extraction, preventing train/validation leakage.

For a very small dataset, set `n_splits` no higher than the number of independent
images or wells.

Run an engineering test first:

```powershell
.\.python311\python.exe scripts\train_instances.py --smoke-test
```

Then train a real fold:

```powershell
.\.python311\python.exe scripts\train_instances.py `
  --config configs/train_instances.yaml `
  --fold 0
```

Training writes:

```text
outputs/checkpoints/astrocyte_instances/best.pt
outputs/checkpoints/astrocyte_instances/last.pt
outputs/checkpoints/astrocyte_instances/history.csv
outputs/checkpoints/astrocyte_instances/cross_validation_assignments.csv
```

## Prediction and evaluation

Predict complete individual cells:

```powershell
.\.python311\python.exe scripts\predict_astrocyte_instances.py `
  --config configs/train_instances.yaml `
  --checkpoint outputs/checkpoints/astrocyte_instances/best.pt `
  --split test
```

Automatic outputs are stored separately from human annotations:

```text
outputs/instance_predictions/raw_heads/       probabilities and ownership offsets
outputs/instance_predictions/labels/          individual cell IDs
outputs/instance_predictions/compartments/    nucleus/soma/process classes
outputs/instance_predictions/overlays/        visual QC
outputs/instance_predictions/cell_measurements.csv
```

Evaluate against held-out human instance masks:

```powershell
.\.python311\python.exe scripts\evaluate_astrocyte_instances.py `
  --manifest outputs/instance_predictions/manifest.csv
```

Instance evaluation reports object precision, recall, F1, matched IoU, panoptic
quality, and process ownership accuracy when explicit compartment truth exists.

## Annotation loop

```text
complete-cell seed annotations
            -> initial training
            -> predictions on unlabeled images
            -> manual process/cell correction
            -> import as corrected or reviewed
            -> grouped retraining
```

This loop allows the project to begin with only a small number of complete-cell
annotations while keeping automatic and human data separate.

## Main scripts

| Script | Purpose |
|---|---|
| `prepare_dataset.py` | Automatic channel selection, nucleus detection, model inputs, and QC |
| `generate_bootstrap_pseudo_labels.py` | Initial binary GFAP proposal |
| `generate_astrocyte_instances.py` | Watershed bootstrap of individual cells |
| `import_instance_annotations.py` | Preserve and validate complete-cell annotations |
| `train_instances.py` | Train compartment, boundary, and ownership heads |
| `predict_astrocyte_instances.py` | Predict complete individual cells |
| `evaluate_astrocyte_instances.py` | Evaluate cell separation and process ownership |
| `generate_pseudo_labels.py` | Generate automatic labels from the retained binary model |
| `select_unlabeled_patches.py` | Rank uncertain patches for correction |
| `import_existing_annotations.py` | Import binary-only annotations |

## Tests

```powershell
.\.python311\python.exe -m pytest -q
```

The tests cover BMP/TIFF loading, nucleus/GFAP preprocessing, patch alignment,
annotation preservation, grouped folds, instance targets, model heads, losses,
ownership offsets, object metrics, notebooks, and dataset behavior.

## Important limitations

- The current `7d_453` result is a watershed bootstrap, not a learned or validated
  process assignment.
- A trained ownership model requires human-corrected complete-cell instance masks.
- DAPI nucleus detection is automatic but classical, and must be inspected for new
  staining, magnification, or acquisition conditions.
- True process crossings can be ambiguous in a two-dimensional image. Z-stacks or
  additional markers may be required when the acquisition contains insufficient
  ownership information.
- Per-cell morphology is only as reliable as the instance and compartment masks.
- No trained checkpoint or biological research dataset is included.

Raw microscopy, external Cellpose files, and original human annotations are never
overwritten by the automatic prediction workflow.
