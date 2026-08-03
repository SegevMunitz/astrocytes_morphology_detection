# Astrocyte Segmentation Pipeline

A reproducible baseline for binary semantic segmentation of GFAP-positive astrocyte
structures in multichannel OME-TIFF microscopy images.

The repository covers data validation, model-input construction, sparse human
annotations, patch-based U-Net training, full-image prediction, grouped
cross-validation, quality control, and preliminary field-level measurements.

> This is research infrastructure, not a scientifically validated astrocyte model.
> Model quality must be established on real held-out experiments before biological use.

## At a glance

The model learns this mapping:

```text
normalized GFAP + nucleus mask + nucleus proximity
                         |
                         v
       background vs GFAP-positive astrocyte structure
```

Current output classes:

```text
0 = background
1 = GFAP-positive astrocyte structure
```

The baseline predicts the complete GFAP-positive field. It does not yet separate
individual astrocytes or distinguish soma from processes.

## Two different types of Cellpose-related masks

This distinction is central to the project:

| Mask | Availability | Role |
|---|---|---|
| Cellpose nucleus instance mask | Required for every image used by the model | Creates two model-input channels |
| Manually corrected astrocyte mask | Available for only a small labeled subset | Supervision target for training/evaluation |

`cellpose_mask_path` in the manifest always refers to the **nucleus instance
mask**. Positive integer values identify individual nuclei; zero is background.

`annotation_path` refers to the **astrocyte segmentation target**. Imported
instance-valued astrocyte masks are archived unchanged and converted to a binary
training mask.

Cellpose execution itself is currently external to this repository. Run nucleus
detection first, preserve the original outputs, and enter their paths in the
manifest.

## How one training sample is constructed

The manifest links the source image, nucleus labels, and human target:

```text
manifest row
  |
  |-- path ------------------> OME-TIFF --> named GFAP channel --> normalization
  |
  |-- cellpose_mask_path ----> nucleus instances
  |                                  |-- labels > 0 --> binary nucleus mask
  |                                  `-- distance --> nucleus proximity map
  |
  `-- annotation_path -------> binary astrocyte target

GFAP + nucleus mask + proximity --> [3, H, W] input
astrocyte target -----------------> [H, W] target
```

The dataset then extracts identical spatial patches from the input and target.
Each returned item contains:

```python
{
    "image": torch.Tensor,        # [3, H, W], float32
    "target": torch.Tensor,       # [H, W], long
    "image_id": str,
    "coordinates": PatchCoordinates,
}
```

### Important implementation detail

Training reads the OME-TIFF, nucleus labels, and annotation paths directly from
the manifest. It constructs the three model channels lazily in the dataset.

The files produced by `extract_channels.py`, `generate_nucleus_inputs.py`, and
`create_patches.py` are useful for inspection, QC, and indexing, but they are not
required inputs for training.

## Repository map

```text
configs/                 YAML settings for data, training, and annotation workflow
data/
  raw/                   immutable source microscopy images
  interim/               optional channel, nucleus, distance, and patch QC outputs
  annotations/
    originals/           archived human-exported instance masks
    binary/              binary human training targets
    qc/                  annotation alignment overlays
  metadata/              manifest and split tables
outputs/
  checkpoints/           model checkpoints and histories
  predictions/           standard model predictions
  pseudo_labels/         automatic labels awaiting human correction
  metrics/               evaluation tables
  feature_tables/        preliminary morphology tables
scripts/                 command-line entry points
src/astroseg/             reusable package implementation
tests/                    synthetic regression tests
notebooks/                empty exploratory notebook templates
```

Within `src/astroseg/`:

| Package | Responsibility |
|---|---|
| `io` | OME-TIFF and manifest loading |
| `preprocessing` | normalization, nucleus conversion, proximity maps, patches |
| `annotations` | non-destructive imports, pseudo-label storage, uncertainty selection |
| `datasets` | manifest filtering and PyTorch sample construction |
| `models` | compact U-Net and model factory |
| `training` | losses, metrics, checkpoints, trainer, grouped folds |
| `inference` | patch prediction and full-image stitching |
| `visualization` | previews, overlays, and QC montages |
| `postprocessing` | preliminary cleanup, skeleton, and nucleus assignment |
| `analysis` | preliminary image and component measurements |

## Manifest

Every microscopy image has one row in `data/metadata/manifest.csv`.

Required project columns:

| Column | Meaning |
|---|---|
| `image_id` | Unique stable image identifier |
| `experiment_id` | Experiment identifier, if known |
| `timepoint` | Experimental time point |
| `treatment` | Treatment label |
| `magnification` | Acquisition magnification |
| `path` | OME-TIFF path |
| `gfap_channel` | Exact OME channel name for GFAP |
| `dapi_channel` | Exact OME channel name for DAPI |
| `cellpose_mask_path` | Nucleus instance-label mask |
| `annotation_path` | Active astrocyte target or pseudo mask |
| `annotation_status` | Annotation lifecycle state |
| `annotation_source` | Human or model provenance |
| `annotator` | Human annotator identifier |
| `review_status` | Free-text review state |
| `split` | `train`, `val`, `test`, or empty |

Additional metadata columns are allowed. For example, add `well_id` to group
cross-validation at well level.

The code does not infer GFAP/DAPI identities or experimental groups from filenames.
Channel names and experimental metadata must be entered explicitly.

## Annotation lifecycle

Allowed `annotation_status` values:

| Status | Meaning | Used for training by default? |
|---|---|---|
| `none` | No astrocyte annotation | No |
| `seed` | Initial manually corrected annotation | Yes |
| `pseudo` | Automatic model prediction awaiting correction | No |
| `corrected` | Human-corrected seed or pseudo label | Yes |
| `reviewed` | Human-reviewed annotation ready for use | Yes |

The default training policy is intentionally conservative:

```yaml
train_annotation_statuses:
  - seed
  - corrected
  - reviewed
```

Pseudo labels cannot enter training accidentally. They must be corrected or
explicitly enabled in configuration.

### Human and automatic data stay separate

Human annotation artifacts:

```text
data/annotations/originals/<image_id>/<status>_<content_hash>.<ext>
data/annotations/binary/<image_id>_<status>_binary.tiff
data/annotations/qc/<image_id>_<status>_annotation_overlay.png
```

Automatic pseudo-label artifacts:

```text
outputs/pseudo_labels/probabilities/<image_id>.npy
outputs/pseudo_labels/masks/<image_id>.tiff
outputs/pseudo_labels/overlays/<image_id>.png
```

Original Cellpose files and imported annotation exports are never modified.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the package and test dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## End-to-end workflow

### 1. Build the initial manifest

```bash
python scripts/build_manifest.py \
  --raw-dir data/raw \
  --output data/metadata/manifest.csv
```

The builder discovers `.tif` and `.tiff` files and assigns stable image IDs. It
does not guess experimental metadata or channel identities. New rows start with
`annotation_status=none`.

After creation, fill at least:

- `gfap_channel`
- `dapi_channel`
- experimental metadata needed for splitting/grouping

### 2. Run Cellpose nucleus detection externally

Run Cellpose on every image that will be used for training or inference. Preserve
the instance-valued nucleus outputs and populate `cellpose_mask_path` for each row.

Expected nucleus mask contract:

```text
2D array aligned with the microscopy image
0 = background
1, 2, 3, ... = nucleus instances
```

### 3. Optional preprocessing and QC

Extract named channels:

```bash
python scripts/extract_channels.py \
  --manifest data/metadata/manifest.csv \
  --output-dir data/interim/channels
```

Generate binary nucleus masks, proximity maps, previews, and montages:

```bash
python scripts/generate_nucleus_inputs.py \
  --manifest data/metadata/manifest.csv \
  --output-dir data/interim
```

Create a coordinate-only patch index without copying image arrays:

```bash
python scripts/create_patches.py \
  --manifest data/metadata/manifest.csv \
  --output data/interim/patches/patch_index.csv \
  --patch-size 512 \
  --overlap 64
```

### 4. Import seed astrocyte annotations

Create a pair table linking image IDs to manually corrected exports:

```csv
image_id,mask_path,annotation_status,annotation_source,annotator,review_status
image_001,exports/image_001_instances.tiff,seed,cellpose_manual_correction,AB,pending
image_002,exports/image_002_instances.tiff,seed,cellpose_manual_correction,AB,pending
```

Import the pairs:

```bash
python scripts/import_existing_annotations.py \
  --manifest data/metadata/manifest.csv \
  --pairs-csv data/metadata/seed_annotation_pairs.csv \
  --output-dir data/annotations \
  --output-manifest data/metadata/manifest_seed.csv \
  --status seed \
  --annotator AB
```

The importer verifies exact dimensions and integer-valued non-negative labels,
archives the original instance mask, creates a binary target, and saves a GFAP
overlay. Always inspect the overlay: equal dimensions alone cannot prove biological
registration when source metadata is incomplete.

### 5. Configure training

Edit `configs/train_binary.yaml` and point `data.manifest_path` to the active
manifest, for example:

```yaml
data:
  manifest_path: data/metadata/manifest_seed.csv
```

Other important defaults:

```yaml
data:
  patch_size: 512
  overlap: 64
  max_nucleus_distance: 64.0

model:
  architecture: unet
  input_channels: 3
  num_classes: 2
  base_channels: 32
```

### 6. Train the baseline

```bash
python scripts/train.py --config configs/train_binary.yaml
```

The trainer uses AdamW, cross-entropy plus Dice loss, early stopping, and saves:

```text
<output.directory>/best.pt
<output.directory>/last.pt
<output.directory>/history.csv
```

Each checkpoint contains model state, optimizer state, epoch, validation metric,
and the complete configuration.

For a quick CPU plumbing check:

```bash
python scripts/train.py --smoke-test --smoke-steps 25
```

### 7. Use grouped cross-validation for the small labeled dataset

Enable cross-validation in the training configuration:

```yaml
cross_validation:
  enabled: true
  n_splits: 5
  validation_fold: 0
  group_column: image_id
  fold_column: fold
```

For well-level grouping, add a non-empty `well_id` manifest column and set:

```yaml
group_column: well_id
```

Run a specific fold:

```bash
python scripts/train.py \
  --config configs/train_binary.yaml \
  --fold 0
```

Fold assignment occurs before patch generation. All patches from one image—and all
images from one well when well grouping is used—remain in the same fold. The run
saves `cross_validation_assignments.csv` beside its checkpoints.

### 8. Evaluate

```bash
python scripts/evaluate.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --split val \
  --output outputs/metrics/validation.csv
```

Evaluation reconstructs each full image before calculating metrics. Overlapping
patch pixels are therefore counted once. The output includes per-image and
image-macro aggregate Dice, IoU, precision, and recall values.

### 9. Predict full images

```bash
python scripts/predict.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --split test \
  --output-dir outputs/predictions
```

For each image, inference:

1. Generates overlapping patches.
2. Applies the model and softmax.
3. Averages probabilities in overlap regions.
4. Restores the original image dimensions.
5. Saves probabilities, a TIFF class mask, and a GFAP overlay.

Hard labels are never stitched directly.

### 10. Generate pseudo labels for unlabeled images

The active manifest in the configuration should contain rows with
`annotation_status=none`.

```bash
python scripts/generate_pseudo_labels.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --output-dir outputs/pseudo_labels \
  --output-manifest data/metadata/manifest_with_pseudo.csv
```

The source manifest is not silently replaced. Matching rows in the new manifest
receive `annotation_status=pseudo` and model provenance.

### 11. Select uncertain patches for correction

```bash
python scripts/select_unlabeled_patches.py \
  --manifest data/metadata/manifest_with_pseudo.csv \
  --probability-dir outputs/pseudo_labels/probabilities \
  --output data/metadata/annotation_queue.csv \
  --patch-size 512 \
  --overlap 64 \
  --top-k 20 \
  --max-patches-per-image 5
```

Patches are ranked by mean normalized predictive entropy. This is a simple active
learning heuristic: high uncertainty indicates useful candidates for human review,
not necessarily biologically important regions.

### 12. Import corrections and retrain

Export corrected masks from the annotation tool and build another pair table. Then
import them against the pseudo manifest:

```bash
python scripts/import_existing_annotations.py \
  --manifest data/metadata/manifest_with_pseudo.csv \
  --pairs-csv data/metadata/corrected_annotation_pairs.csv \
  --output-dir data/annotations \
  --output-manifest data/metadata/manifest_corrected.csv \
  --status corrected \
  --annotator AB \
  --overwrite
```

`--overwrite` permits the active annotation lifecycle entry to advance. Archived
content-addressed originals are still never overwritten.

Point `data.manifest_path` to `manifest_corrected.csv` and repeat training.

### 13. Extract preliminary field-level features

```bash
python scripts/extract_features.py \
  --mask-dir outputs/predictions/masks \
  --output outputs/feature_tables/test_features.csv
```

Current measurements include positive pixels, positive area fraction, connected
components, skeleton pixels, branch pixels, and endpoints. They are preliminary
pixel-level field summaries, not validated single-cell morphology measurements.

## Model and training details

### Baseline model

The implemented model is a compact 2D U-Net with three encoder levels, a
bottleneck, skip-connected decoder levels, and a final `1x1` convolution. It
returns per-pixel logits:

```text
[B, num_classes, H, W]
```

Odd image dimensions are handled by interpolation at skip connections. There is
no fully connected classification head.

### Loss

Training combines:

- categorical cross-entropy for per-pixel classification;
- soft multiclass Dice loss for region overlap.

Background is excluded from Dice by default.

### Metrics

The evaluation API reports per-class and foreground-macro:

- Dice;
- intersection over union;
- precision;
- recall.

Empty-class cases follow explicit finite rules and never return NaN.

## Command reference

| Script | Purpose |
|---|---|
| `build_manifest.py` | Discover TIFFs and create a conservative manifest template |
| `extract_channels.py` | Save explicit GFAP/DAPI arrays and previews |
| `generate_nucleus_inputs.py` | Validate Cellpose nuclei and generate masks, proximity, QC |
| `create_patches.py` | Create a coordinate-only patch index |
| `import_existing_annotations.py` | Archive, validate, binarize, and record human masks |
| `train.py` | Train U-Net, run grouped folds, or execute the smoke test |
| `evaluate.py` | Reconstruct full annotated images and save metrics |
| `predict.py` | Run full-image inference for a manifest split |
| `generate_pseudo_labels.py` | Predict `none` rows and create a pseudo-label manifest |
| `select_unlabeled_patches.py` | Rank patches by predictive entropy |
| `extract_features.py` | Extract preliminary mask and skeleton features |

Every script provides detailed arguments through:

```bash
python scripts/<script_name>.py --help
```

## Recommended code-reading order

To understand the implementation without reading every file, follow one sample:

1. `configs/train_binary.yaml` — experiment settings.
2. `scripts/train.py` — training entry point.
3. `src/astroseg/io/manifest.py` — manifest contract.
4. `src/astroseg/datasets/astrocyte_dataset.py` — input/target construction.
5. `src/astroseg/preprocessing/` — channel transformations and patches.
6. `src/astroseg/models/unet.py` — model forward pass.
7. `src/astroseg/training/trainer.py` — optimization and checkpoints.
8. `src/astroseg/inference/predict_full_image.py` — full-resolution prediction.
9. `src/astroseg/annotations/` — sparse annotation lifecycle.

All functions and classes include concise multi-line docstrings describing their
role, data contract, and important constraints.

## Tests

Tests use synthetic images and temporary directories; no research data is needed.

Coverage includes:

- OME-TIFF axes, dtype, metadata, and channel selection;
- normalization and nucleus validation;
- proximity-map constraints;
- patch coverage and exact stitching;
- dataset filtering, shapes, and target class ranges;
- odd-sized U-Net output;
- Dice, IoU, precision, recall, and empty classes;
- full-image metric reconstruction;
- non-destructive annotation import;
- pseudo-label storage and uncertainty selection;
- image/well grouped cross-validation and fold balancing.

Run:

```bash
pytest
```

## Current limitations

- Cellpose is not executed by this repository; nucleus masks are external inputs.
- The scientifically supported task is binary segmentation only.
- Multiclass configuration exists for future work, but multiclass annotations are
  not yet available.
- SegFormer is an explicit `NotImplementedError` placeholder.
- Notebooks are empty exploratory templates.
- Prediction is patch-by-patch rather than batched for throughput.
- Checkpoints contain optimizer state, but no resume-training CLI is implemented.
- Skeleton, branch, endpoint, and nucleus-assignment outputs are preliminary.
- No model in this repository should be considered biologically validated yet.

## Development principles

- Keep raw microscopy and original annotation exports immutable.
- Never guess missing channel identities.
- Keep image I/O separate from preprocessing and model code.
- Define image/well splits before patch generation.
- Keep automatic predictions separate from human-reviewed annotations.
- Store complete configuration and fold assignments with training outputs.
- Fail clearly on invalid dimensions, labels, classes, or missing files.
- Prefer a small reproducible baseline over an unverified complex model.
