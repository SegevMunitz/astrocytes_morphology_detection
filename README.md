# Astrocyte Segmentation Pipeline

A research-oriented image-analysis project for detecting and quantifying GFAP-positive astrocyte structures in multichannel microscopy images.

The first goal is to build a reproducible baseline pipeline that:

1. Reads multichannel OME-TIFF microscopy files.
2. Extracts the GFAP and DAPI channels.
3. Loads Cellpose nucleus-instance masks.
4. Converts nucleus labels into model-ready inputs.
5. Trains a semantic-segmentation model for GFAP-positive astrocyte structures.
6. Produces full-image predictions, overlays, quality-control outputs, and quantitative measurements.

The initial model is a binary semantic-segmentation model:

- `0`: background
- `1`: GFAP-positive astrocyte structure

The project should later support multiclass segmentation:

- `0`: background
- `1`: astrocyte soma
- `2`: astrocyte processes

## Scientific motivation

Cellpose detects nuclei reliably in the current dataset, but it does not adequately segment complete astrocytes, especially thin GFAP-positive processes.

This project therefore treats detected nuclei as spatial anchors and learns astrocyte segmentation from:

- GFAP fluorescence
- nucleus mask
- nucleus-distance map

The initial focus is segmentation of the full GFAP-positive astrocyte network rather than perfect assignment of every process to an individual astrocyte.

## Initial model input and output

### Input channels

Each training sample should contain three channels:

1. GFAP image
2. Binary nucleus mask derived from Cellpose labels
3. Nucleus-distance map

Tensor shape:

```text
[C, H, W] = [3, H, W]
```

### Output

For the initial binary model:

```text
[H, W]
```

with class labels:

```text
0 = background
1 = GFAP-positive astrocyte structure
```

The model itself should output logits of shape:

```text
[num_classes, H, W]
```

For binary segmentation with two explicit classes:

```text
[2, H, W]
```

## Recommended project structure

```text
astrocyte-segmentation/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── train_binary.yaml
│   └── train_multiclass.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   │   ├── channels/
│   │   ├── nucleus_masks/
│   │   ├── nucleus_distance_maps/
│   │   └── patches/
│   ├── annotations/
│   │   ├── binary/
│   │   ├── multiclass/
│   │   └── ignore_masks/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/
│       ├── manifest.csv
│       └── splits.csv
│
├── notebooks/
│   ├── 01_inspect_tiff_channels.ipynb
│   ├── 02_visualize_cellpose_masks.ipynb
│   ├── 03_create_initial_annotations.ipynb
│   └── 04_evaluate_baseline.ipynb
│
├── src/
│   └── astroseg/
│       ├── __init__.py
│       ├── constants.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── ome_tiff.py
│       │   └── manifest.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── normalize.py
│       │   ├── nuclei.py
│       │   ├── distance_maps.py
│       │   └── patches.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── astrocyte_dataset.py
│       │   └── augmentations.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── unet.py
│       │   ├── segformer.py
│       │   └── model_factory.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py
│       │   ├── metrics.py
│       │   ├── trainer.py
│       │   └── checkpoints.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predict_patch.py
│       │   ├── predict_full_image.py
│       │   └── stitch.py
│       ├── postprocessing/
│       │   ├── __init__.py
│       │   ├── clean_masks.py
│       │   ├── skeleton.py
│       │   └── nucleus_assignment.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── morphology.py
│       │   ├── image_features.py
│       │   └── aggregate.py
│       └── visualization/
│           ├── __init__.py
│           ├── overlays.py
│           └── qc_plots.py
│
├── scripts/
│   ├── build_manifest.py
│   ├── extract_channels.py
│   ├── generate_nucleus_inputs.py
│   ├── create_patches.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── extract_features.py
│
├── tests/
│   ├── test_ome_tiff.py
│   ├── test_patches.py
│   ├── test_dataset.py
│   └── test_metrics.py
│
└── outputs/
    ├── checkpoints/
    ├── predictions/
    ├── overlays/
    ├── metrics/
    └── feature_tables/
```

## Pipeline overview

```text
OME-TIFF
   ↓
Read image and metadata
   ↓
Identify GFAP and DAPI channels
   ↓
Load Cellpose nucleus labels
   ↓
Create binary nucleus mask
   ↓
Create nucleus-distance map
   ↓
Normalize model inputs
   ↓
Split into overlapping patches
   ↓
Train segmentation model
   ↓
Predict full-resolution masks
   ↓
Stitch overlapping predictions
   ↓
Post-process and create overlays
   ↓
Extract quantitative features
```

## Data organization

Raw data should remain unchanged.

Recommended convention:

```text
data/raw/<experiment_id>/<original_file_name>.tif
```

Cellpose masks should be stored separately:

```text
data/interim/nucleus_masks/<image_id>_nuclei.tif
```

Annotations should be stored using the same `image_id`:

```text
data/annotations/originals/<image_id>/<status>_<content_hash>.<ext>
data/annotations/binary/<image_id>_<status>_binary.tiff
data/annotations/qc/<image_id>_<status>_annotation_overlay.png
```

Automatic predictions and pseudo labels are stored separately:

```text
outputs/pseudo_labels/probabilities/<image_id>.npy
outputs/pseudo_labels/masks/<image_id>.tiff
outputs/pseudo_labels/overlays/<image_id>.png
```

Never overwrite raw microscopy images, original Cellpose files, or imported
annotation exports. Imported masks are archived by content hash before a binary
training target is created.

## Manifest

Every image should have one row in `data/metadata/manifest.csv`.

Recommended fields:

```text
image_id
experiment_id
culture_date
staining_date
plate
well
field
timepoint
treatment
magnification
biological_replicate
technical_replicate
path
gfap_channel
dapi_channel
has_brightfield
cellpose_mask_path
annotation_path
annotation_status
annotation_source
annotator
review_status
split
```

`annotation_status` is required and must be one of:

- `none`: no annotation is available
- `seed`: manually corrected seed annotation
- `pseudo`: automatic model prediction awaiting correction
- `corrected`: a human corrected a seed or pseudo label
- `reviewed`: a human-reviewed annotation ready for use

By default, datasets train only on `seed`, `corrected`, and `reviewed` rows.
Pseudo labels never enter training unless the configuration explicitly opts in.

The code should not infer experimental groups from filenames during training. Filename parsing may be used to build the initial manifest, but all downstream code should rely on the manifest.

## Train, validation, and test splitting

Do not split random patches from the same image across train and test.

Splits must be defined before patch extraction and should be grouped by the highest available biological level, such as:

- biological replicate
- experiment
- plate
- well

The test set should contain images from wells or experiments not seen during training.

Store the split assignments in:

```text
data/metadata/splits.csv
```

For very small labeled datasets, grouped cross-validation can instead be enabled
in the training configuration. Fold assignment happens at the manifest image or
well level before patches are generated, so patches from one image cannot appear
in different folds. Use `group_column: image_id` by default, or add an explicit
non-empty `well_id` column and use `group_column: well_id`.

Each cross-validation run also saves `cross_validation_assignments.csv` beside its
checkpoints, recording the image/group, fold, and effective split used by that run.
Evaluation reconstructs complete probability maps before calculating metrics, so
pixels in overlapping patches are counted exactly once.

## Seed annotation and correction workflow

The supported iterative workflow is:

```text
manually corrected Cellpose seed masks
    -> initial binary U-Net training
    -> full-image predictions on annotation_status=none images
    -> uncertainty-ranked patch selection
    -> manual correction and review
    -> import corrected masks
    -> retraining on seed + corrected + reviewed masks
```

An import pair table minimally contains:

```csv
image_id,mask_path,annotation_status,annotation_source,annotator,review_status
image_001,exports/image_001_instances.tiff,seed,cellpose_manual_correction,AB,pending
```

Image identity, exact dimensions, numeric non-negative labels, and the named GFAP
channel are validated. The generated overlay must still be inspected because
dimensions alone cannot prove biological registration when source TIFF metadata
is incomplete. Instance IDs are converted to a binary target (`0` background,
`1` GFAP-positive structure) while the instance-valued source is retained.

## Preprocessing

### GFAP normalization

Use percentile normalization for model input only.

Example:

```text
lower percentile = 1.0
upper percentile = 99.8
```

Do not use independently normalized images for biological intensity comparisons.

Maintain:

1. normalized images for model training
2. minimally processed raw-intensity images for quantitative analysis

### Nucleus mask

Cellpose nucleus labels are expected to be instance-label images:

```text
0 = background
1, 2, 3, ... = individual nuclei
```

Convert them into a binary mask:

```text
0 = no nucleus
1 = nucleus
```

### Nucleus-distance map

Create a map that is high near nuclei and decreases with distance.

One suitable definition is:

```text
proximity = 1 - min(distance, max_distance) / max_distance
```

The output should be a float image in `[0, 1]`.

## Patch extraction

Initial recommendation:

```text
patch size = 512 × 512
overlap = 64 pixels
```

Every patch must retain:

- `image_id`
- top-left `x`
- top-left `y`
- width
- height
- split

Patch extraction must be deterministic and testable.

Avoid saving patches unless required. Prefer lazy patch extraction from full images when practical.

## Baseline model

The first baseline should be a small 2D U-Net implemented in PyTorch.

Requirements:

- configurable number of input channels
- configurable number of output classes
- skip connections
- no fully connected classification head
- output spatial resolution equal to input resolution

Initial configuration:

```yaml
model:
  architecture: unet
  input_channels: 3
  num_classes: 2
  base_channels: 32
```

The code should support adding SegFormer later through a model factory.

## Loss

Initial loss:

```text
total_loss = cross_entropy + dice_loss
```

Do not add advanced losses until the basic pipeline is verified.

Later candidates:

- focal loss
- class-weighted cross-entropy
- clDice for thin-process topology

## Metrics

Initial metrics:

- Dice score
- Intersection over Union
- precision
- recall

Later topology-aware metrics:

- clDice
- skeleton-length error
- branch-point error
- endpoint error

Metrics should be reported:

- globally
- per image
- per experimental timepoint
- per test experiment or well

## Training configuration

Example:

```yaml
seed: 42

data:
  manifest_path: data/metadata/manifest.csv
  splits_path: data/metadata/splits.csv
  patch_size: 512
  overlap: 64
  num_workers: 4
  train_annotation_statuses:
    - seed
    - corrected
    - reviewed

model:
  architecture: unet
  input_channels: 3
  num_classes: 2
  base_channels: 32

training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 15

loss:
  cross_entropy_weight: 1.0
  dice_weight: 1.0

cross_validation:
  enabled: false
  n_splits: 5
  validation_fold: 0
  group_column: image_id
  fold_column: fold

output:
  directory: outputs/checkpoints/binary_baseline
```

## Inference

Full-image inference should:

1. generate overlapping patches
2. run the model on each patch
3. convert logits to probabilities
4. combine overlapping probability maps
5. use weighted averaging in overlap regions
6. apply `argmax` to create the final mask
7. save:
   - probability map
   - class mask
   - overlay image
   - metadata JSON

Avoid stitching hard class labels directly.

## Quality control

For each image, create a QC montage containing:

- raw GFAP
- DAPI
- Cellpose nucleus labels
- binary nucleus mask
- nucleus-distance map
- annotation, when available
- model prediction
- overlay

The first development milestone is:

> Given one OME-TIFF and its Cellpose mask, generate a correctly aligned three-channel model input and a QC montage.

## Feature extraction

After segmentation is stable, extract image-level features such as:

- GFAP-positive area
- GFAP-positive fraction
- total GFAP intensity
- mean GFAP intensity
- number of nuclei
- GFAP area per nucleus
- skeleton length
- branch points
- endpoints
- connected components

Do not treat individual cells from the same well as independent biological replicates.

Features should eventually be summarized at:

- field level
- well level
- biological-replicate level

## Installation

Recommended Python version:

```text
Python 3.11
```

Create and activate a Python 3.11 environment, then install the package and
development dependencies in editable mode:

```bash
pip install -e ".[dev]"
```

Core dependencies:

```text
numpy
pandas
scipy
scikit-image
tifffile
ome-types
matplotlib
pyyaml
pydantic
torch
torchvision
albumentations
tqdm
pytest
```

Optional later dependencies:

```text
segmentation-models-pytorch
transformers
monai
cellpose
```

## Expected command-line workflow

```bash
python scripts/build_manifest.py \
  --raw-dir data/raw \
  --output data/metadata/manifest.csv
```

```bash
python scripts/extract_channels.py \
  --manifest data/metadata/manifest.csv \
  --output-dir data/interim/channels
```

```bash
python scripts/generate_nucleus_inputs.py \
  --manifest data/metadata/manifest.csv \
  --output-dir data/interim
```

Import initial manually corrected Cellpose astrocyte masks. The input exports are
copied unchanged before binary masks are created:

```bash
python scripts/import_existing_annotations.py \
  --manifest data/metadata/manifest.csv \
  --pairs-csv data/metadata/seed_annotation_pairs.csv \
  --output-dir data/annotations \
  --output-manifest data/metadata/manifest_seed.csv \
  --status seed \
  --annotator AB
```

Point `data.manifest_path` in the training configuration at the seed manifest,
then train normally or choose a grouped validation fold:

```bash
python scripts/train.py \
  --config configs/train_binary.yaml
```

```bash
python scripts/train.py \
  --config configs/train_binary.yaml \
  --fold 0
```

Generate automatic labels only for rows in state `none`. This writes a new
manifest and keeps automatic files under `outputs/`:

```bash
python scripts/generate_pseudo_labels.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --output-dir outputs/pseudo_labels \
  --output-manifest data/metadata/manifest_with_pseudo.csv
```

Rank uncertain pseudo-labeled patches for manual correction:

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

After correction, import the new masks with `--status corrected` and a new output
manifest, update `data.manifest_path`, and rerun training. `--overwrite` permits a
new lifecycle state to replace the active binary target for an image; archived
content-addressed originals are never overwritten.

```bash
python scripts/evaluate.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt
```

```bash
python scripts/predict.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --split test
```

## Development principles

- Keep raw data immutable.
- Keep notebooks exploratory; production logic belongs in `src/astroseg`.
- Add type hints and docstrings to public functions.
- Use deterministic random seeds.
- Add tests for shape, dtype, alignment, and patch stitching.
- Fail clearly when expected channels or masks are missing.
- Do not silently guess channel identities.
- Store training configuration with every checkpoint.
- Save enough metadata to reproduce every result.
- Prefer a simple working baseline over a complex unverified architecture.

## Initial milestone checklist

- [ ] Project installs successfully with `pip install -e ".[dev]"`
- [ ] OME-TIFF loader reads image and channel metadata
- [ ] Manifest builder catalogs raw files
- [ ] Cellpose labels align with microscopy images
- [ ] Binary nucleus masks are generated
- [ ] Nucleus-distance maps are generated
- [ ] QC montage is generated for one full image
- [ ] Patch extraction and stitching pass tests
- [ ] PyTorch dataset returns valid tensors
- [ ] Small U-Net trains on a tiny sample
- [ ] Overfit test succeeds on one or two patches
- [ ] Full-image prediction and overlay are saved
- [ ] Dice and IoU are reported
