# Astrocyte Segmentation Pipeline

A research-oriented image-analysis project for segmenting GFAP-positive astrocyte
structures in multichannel microscopy TIFF and OME-TIFF images.

The repository implements the complete baseline path from microscopy files and
automatically detected nucleus labels to patch-based U-Net training, full-resolution prediction,
quality-control images, grouped cross-validation, and preliminary measurements.
It also supports a small seed dataset of manually corrected astrocyte masks and an
iterative annotation workflow for expanding that dataset safely.

The current task is **binary semantic segmentation**:

```text
0 = background
1 = GFAP-positive astrocyte structure
```

The output is one foreground mask for the complete GFAP-positive field. The model
does not currently identify individual astrocytes or distinguish soma from
processes.

## Scientific motivation

The DAPI channel provides a strong signal for nucleus detection, but a nucleus mask
is not a complete astrocyte mask. Thin GFAP-positive processes can extend far
from the nucleus, overlap with neighboring structures, and are often not captured
well by an instance-segmentation method focused on compact objects.

This project therefore uses nuclei as spatial context rather than as the answer.
For every pixel, the model combines:

1. GFAP fluorescence, which contains the structure to segment.
2. A binary nucleus mask, derived from internal or externally supplied instance labels.
3. A smooth nucleus-proximity map, which tells the model how close the pixel is
   to the nearest detected nucleus.

The model learns the astrocyte target from a small number of human-corrected masks.
As more masks are corrected and reviewed, they can be added to the training set
without changing the basic pipeline.

Future multiclass work may separate background, soma, and processes, but the
implemented and scientifically intended baseline is binary.

## The project in one diagram

```text
microscopy TIFF
    |
    |-- automatic GFAP channel -----------> percentile normalization ------+
    |                                                                      |
    |                                                                      v
    |-- automatic DAPI channel --> classical Otsu + watershed               |
    |                                      (not a neural network)            |
    |                                                   --> nucleus instances|
                                                  |-- binary nucleus mask ->|
                                                  `-- proximity map --------+
                                                                           |
                                                                           v
Human astrocyte mask --> validated binary target [H, W] ------------> 2D U-Net
                                                                           |
                                                                           v
                                                        background / astrocyte
                                                                           |
                                        +----------------------------------+
                                        | probabilities, mask, overlay     |
                                        | metrics and preliminary features |
                                        +----------------------------------+
```

The manifest is the index connecting all these files. One manifest row describes
one microscopy image, its channel names, its nucleus labels, its annotation state,
and its train/validation/test assignment.

## The two masks must not be confused

There are two different segmentation-like inputs in this project:

| File | Manifest column | What it contains | How it is used |
|---|---|---|---|
| Nucleus instance mask | `cellpose_mask_path` (legacy column name) | `0` for background and positive instance IDs for nuclei | Converted into two model-input channels |
| Astrocyte annotation | `annotation_path` | Binary GFAP-positive target used as ground truth | Used only for training and evaluation |

The nucleus instance mask is needed for **every image passed through the model**,
including unlabeled images used only for prediction. An astrocyte annotation is
needed only when the image is used for supervised training or evaluation.

This means that it is normal to have:

- nucleus masks for all usable images;
- manually corrected astrocyte annotations for only a small subset;
- no astrocyte annotation yet for most images.

`prepare_dataset.py` detects nuclei internally from DAPI and writes the resulting
instance-label path to `cellpose_mask_path`. The column keeps its historical name
for compatibility with the existing dataset and training code. External Cellpose
labels remain supported as an optional alternative.

### Files required by each operation

| Operation | Microscopy TIFF | GFAP channel | Nucleus instance mask | Astrocyte annotation |
|---|---:|---:|---:|---:|
| Optional channel/QC export | Yes | Yes | Only for nucleus QC | No |
| Supervised training | Yes | Yes | Yes | Yes |
| Evaluation | Yes | Yes | Yes | Yes |
| Standard prediction | Yes | Yes | Yes | No |
| Pseudo-label generation | Yes | Yes | Yes | No |
| Annotation import | Yes | Yes | No | Imported mask is the input |

The DAPI channel is useful for inspection and QC, but it is not one of the three
current model-input channels.

## What one model sample contains

For one complete image, the input array has this channel-first shape:

```text
[C, H, W] = [3, H, W]
```

| Channel | Construction | Range | Purpose |
|---|---|---:|---|
| `0`: GFAP | Named OME channel, percentile-normalized | `[0, 1]` | Main appearance signal |
| `1`: nucleus mask | `cellpose_labels > 0` | `{0, 1}` | Exact detected-nucleus locations |
| `2`: nucleus proximity | Inverted, clipped Euclidean distance | `[0, 1]` | Smooth distance-to-nucleus context |

The target is an integer array:

```text
[H, W]
```

The U-Net returns logits:

```text
[B, 2, H, W]
```

The dataset extracts matching spatial patches from the three-channel input and
the target. One item returned by `AstrocyteDataset` is:

```python
{
    "image": torch.Tensor,          # [3, patch_H, patch_W], float32
    "target": torch.Tensor,         # [patch_H, patch_W], int64
    "image_id": str,                # original manifest identity
    "coordinates": PatchCoordinates # y, x, height, width in the full image
}
```

Training constructs these arrays lazily from the paths in the manifest. The
intermediate files generated by `prepare_dataset.py`, `extract_channels.py`,
`generate_nucleus_inputs.py`, and `create_patches.py` are useful for inspection,
QC, and debugging. Training reads the generated instance-label path from the
manifest but constructs the binary mask and proximity channel lazily.

## Repository structure

```text
astrocytes_morphology_detection/
|
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- data.yaml
|   |-- train_binary.yaml
|   |-- train_multiclass.yaml
|   `-- annotation_workflow.yaml
|
|-- data/
|   |-- raw/                         original microscopy images
|   |-- interim/                     optional preprocessing and QC outputs
|   |-- annotations/
|   |   |-- originals/               archived human-exported masks
|   |   |-- binary/                  derived binary training targets
|   |   `-- qc/                      annotation overlays
|   `-- metadata/                    manifests, pair tables, patch queues
|
|-- scripts/                         command-line entry points
|-- src/astroseg/                    reusable implementation
|   |-- io/                          OME-TIFF and manifest handling
|   |-- preprocessing/               normalization, nuclei, distance, patches
|   |-- annotations/                 imports, pseudo labels, patch selection
|   |-- datasets/                    PyTorch dataset and augmentation
|   |-- models/                      U-Net and model factory
|   |-- training/                    trainer, losses, metrics, folds, checkpoints
|   |-- inference/                   patch prediction and full-image stitching
|   |-- visualization/               previews, overlays, QC montages
|   |-- postprocessing/              mask cleanup and exploratory cell assignment
|   `-- analysis/                    field and component measurements
|
|-- tests/                           synthetic regression tests
|-- notebooks/                       space for exploratory notebooks
`-- outputs/
    |-- checkpoints/                 trained model states and histories
    |-- predictions/                 normal model outputs
    |-- pseudo_labels/               automatic labels awaiting correction
    |-- metrics/                     evaluation CSV files
    `-- feature_tables/              preliminary quantitative summaries
```

### Main Python modules

| Module | Main API | Responsibility |
|---|---|---|
| `io/ome_tiff.py` | `load_ome_tiff`, `get_channel` | Loads OME-TIFF data, normalizes axes to channel-first form, reads channel names and pixel size, and retrieves channels explicitly by name. |
| `io/manifest.py` | `ManifestRow`, `load_manifest` | Defines and validates the one-row-per-image data contract. It rejects missing columns, duplicate IDs, invalid states, and annotated rows without an annotation path. |
| `preprocessing/channels.py` | `select_model_channels` | Uses explicit metadata when available and automatically maps RGB composites to Blue DAPI plus the stronger Red/Green structural channel. |
| `preprocessing/nucleus_detection.py` | `detect_nucleus_instances` | Detects bright DAPI nuclei with percentile normalization, Gaussian smoothing, Otsu thresholding, and marker watershed. |
| `preprocessing/normalize.py` | `percentile_normalize` | Scales GFAP intensities to `[0, 1]` for model input. The result is not intended for biological intensity comparisons. |
| `preprocessing/nuclei.py` | `validate_nucleus_labels`, `labels_to_binary_mask` | Checks exact image alignment and valid instance labels, then converts all positive nucleus IDs to foreground. |
| `preprocessing/distance_maps.py` | `create_nucleus_proximity_map` | Converts the binary nucleus mask into a bounded distance-derived context channel. |
| `preprocessing/patches.py` | `PatchCoordinates`, `generate_patch_coordinates`, `stitch_probability_patches` | Creates deterministic, border-covering patch coordinates and reconstructs complete probability maps by averaging overlaps. |
| `datasets/astrocyte_dataset.py` | `prepare_model_inputs`, `AstrocyteDataset` | Filters manifest rows, validates every referenced file, builds the three channels, and returns aligned PyTorch patches. |
| `models/unet.py` | `UNet` | Implements the compact encoder-decoder baseline with skip connections and equal input/output spatial size. |
| `training/losses.py` | `DiceLoss`, `CrossEntropyDiceLoss` | Combines categorical cross-entropy with soft Dice overlap loss. Background is excluded from Dice by default. |
| `training/trainer.py` | `train_model`, `run_overfit_smoke_test` | Runs AdamW optimization, validation, early stopping, checkpointing, history export, and a synthetic end-to-end diagnostic. |
| `training/cross_validation.py` | `assign_grouped_folds`, `load_grouped_fold_manifests` | Assigns entire images or wells to deterministic folds and checks that no group leaks across train and validation. |
| `training/metrics.py` | `metrics_from_predictions`, `metrics_from_probability_patches` | Computes finite Dice, IoU, precision, and recall after reconstructing complete images. |
| `inference/predict_full_image.py` | `predict_full_image` | Predicts overlapping patches and averages softmax probabilities back into the original image dimensions. |
| `annotations/workflow.py` | `import_annotation_pair` | Validates a human mask, archives its original bytes, converts it to binary, and produces a QC overlay and provenance record. |
| `annotations/pseudo_labels.py` | `save_pseudo_label_artifacts` | Stores automatic probabilities, masks, and overlays under `outputs/`, separate from human annotations. |
| `annotations/selection.py` | `select_uncertain_patches` | Scores candidate patches using normalized predictive entropy and creates a ranked correction queue. |
| `analysis/image_features.py` | `extract_image_features` | Measures preliminary field-level area, connected components, skeleton length, branches, and endpoints. |

The files in `scripts/` connect configuration and file paths to these reusable
modules. Business logic should remain in `src/astroseg/`; scripts should remain
small command-line entry points.

## Data organization and preservation rules

Raw data should remain unchanged. A typical source layout is:

```text
data/raw/<experiment_id>/<original_file_name>.ome.tif
```

Automatically detected nucleus labels are stored as:

```text
data/interim/nucleus_labels/<image_id>_nuclei.tiff
```

Imported human annotations are stored as:

```text
data/annotations/originals/<image_id>/<status>_<content_hash>.<ext>
data/annotations/binary/<image_id>_<status>_binary.tiff
data/annotations/qc/<image_id>_<status>_annotation_overlay.png
```

Automatic pseudo labels are stored separately:

```text
outputs/pseudo_labels/probabilities/<image_id>.npy
outputs/pseudo_labels/masks/<image_id>.tiff
outputs/pseudo_labels/overlays/<image_id>.png
```

The project follows four preservation rules:

1. Never modify raw microscopy images.
2. Regenerate only derived internal masks; never modify optional external Cellpose files.
3. Archive imported human exports before deriving binary masks.
4. Never store automatic predictions in the human-annotation directories.

## Manifest: the central project index

Every microscopy image has one row in a CSV manifest, normally
`data/metadata/manifest.csv`.

### Required schema

| Column | Meaning | When it must be populated |
|---|---|---|
| `image_id` | Unique, stable image identifier | Always |
| `experiment_id` | Experiment identifier | When known; useful for analysis/grouping |
| `timepoint` | Experimental time point | When known |
| `treatment` | Experimental treatment | When known |
| `magnification` | Acquisition magnification | When known |
| `path` | Source OME-TIFF path | Always |
| `gfap_channel` | Exact GFAP channel name in OME metadata | Before preprocessing, training, or prediction |
| `dapi_channel` | Exact DAPI channel name | Before DAPI/QC export |
| `cellpose_mask_path` | Active nucleus instance-label file; name retained for compatibility | Filled automatically by `prepare_dataset.py` |
| `annotation_path` | Active binary astrocyte target or pseudo mask | Required when status is not `none` |
| `annotation_status` | `none`, `seed`, `pseudo`, `corrected`, or `reviewed` | Always |
| `annotation_source` | Human or model provenance | When an annotation exists |
| `annotator` | Human annotator identifier | For human annotations when available |
| `review_status` | Review state such as `pending` or `approved` | When applicable |
| `split` | `train`, `val`, `test`, or empty | Before normal training/evaluation |

Extra columns are allowed. Useful examples include `plate`, `well_id`, `field`,
`biological_replicate`, and `technical_replicate`. These fields are important for
preventing biological leakage and for later aggregation.

Example:

```csv
image_id,experiment_id,timepoint,treatment,magnification,path,gfap_channel,dapi_channel,cellpose_mask_path,annotation_path,annotation_status,annotation_source,annotator,review_status,split,well_id
img_001,exp_A,day7,control,20x,data/raw/exp_A/img_001.ome.tif,GFAP,DAPI,data/interim/nucleus_labels/img_001.tiff,data/annotations/binary/img_001_seed_binary.tiff,seed,manual_cellpose_correction,AB,pending,train,A01
img_002,exp_A,day7,control,20x,data/raw/exp_A/img_002.ome.tif,GFAP,DAPI,data/interim/nucleus_labels/img_002.tiff,,none,,,,test,A02
```

The code deliberately does not guess GFAP/DAPI identities, treatments, or groups
from filenames. Fill these values explicitly after building the initial manifest.

### Path resolution

Manifest paths may be absolute or relative. For relative values, the code checks
the path as written and then relative to the directory containing the manifest.
Keeping paths project-relative is usually the most portable choice.

## Annotation lifecycle

The annotation status records what kind of astrocyte target is currently active
for an image:

| Status | Meaning | Used for training by default? |
|---|---|---:|
| `none` | No astrocyte annotation exists | No |
| `seed` | Initial manually corrected annotation | Yes |
| `pseudo` | Automatic model prediction awaiting correction | No |
| `corrected` | A human corrected a seed or pseudo label | Yes |
| `reviewed` | A human reviewed the annotation and accepted it | Yes |

The intended lifecycle is:

```text
none --> seed -----------------------------> reviewed
  |                                           ^
  `--> pseudo --> human correction --> corrected
```

The default training configuration accepts only:

```yaml
train_annotation_statuses:
  - seed
  - corrected
  - reviewed
```

Therefore, generating a pseudo label does not silently add it to training. It must
first be corrected/reviewed, or the user must explicitly change the configuration.

### What annotation import validates

`import_existing_annotations.py` checks that:

- the image ID exists exactly once in the manifest;
- the OME-TIFF and mask paths exist;
- the selected GFAP channel exists;
- the mask is a two-dimensional numeric array;
- image and mask dimensions are identical;
- mask labels are finite, non-negative, and integer-valued;
- existing derived artifacts are not overwritten unless requested.

Dimension equality is a necessary alignment check, but it cannot prove biological
registration if the exported mask lacks complete spatial metadata. Always inspect
the generated GFAP overlay.

Instance-valued astrocyte masks are converted using `mask > 0`. The original
instance-valued file is archived unchanged using a content hash, so instance IDs
remain available for later work even though the current target is binary.

## Preprocessing in detail

### Microscopy TIFF loading

`load_ome_tiff` reads the image and OME metadata, normalizes supported layouts to
channel-first `[C, H, W]`, and records channel names and physical pixel size when
available. Unsupported dimensions or ambiguous channel requests fail explicitly.

Standard non-OME RGB TIFFs with `YXS` axes are also supported. Because their color
samples have no biological metadata, the loader exposes them as `Red`, `Green`,
and `Blue`; the manifest must still state which color represents GFAP and DAPI for
each image. For example, use `gfap_channel=Green, dapi_channel=Blue` for a
green/blue composite and `gfap_channel=Red, dapi_channel=Blue` for a red/blue
composite.

Use exact OME channel names in the manifest. The loader does not assume that a
fixed channel index is always GFAP.

### GFAP normalization

The model uses percentile normalization with defaults:

```text
lower percentile = 1.0
upper percentile = 99.8
```

If `p_low` and `p_high` are those image percentiles, the normalized value is:

```text
normalized = clip((image - p_low) / (p_high - p_low), 0, 1)
```

This reduces the influence of extreme pixels and produces a stable model-input
range. It must not be used as a replacement for raw intensity values in biological
comparisons across images or experiments.

Maintain two conceptual data paths:

1. normalized images for model input;
2. raw or minimally processed intensities for quantitative fluorescence analysis.

### Automatic nucleus instance detection

`detect_nucleus_instances` runs directly on the selected DAPI channel. It does not
require a separately trained neural network or nucleus annotations. The default
pipeline is:

1. percentile-normalize DAPI to `[0, 1]`;
2. apply Gaussian smoothing (`sigma=1.2`);
3. calculate an Otsu foreground threshold;
4. remove foreground components smaller than 30 pixels;
5. fill holes inside detected nuclei;
6. find distance-transform peaks at least 7 pixels apart;
7. use marker watershed to split touching nuclei;
8. save sequential `uint32` instance labels.

These defaults produced 341 instances on `7d_453.tif`. They are configurable from
the command line and must be checked on additional acquisition conditions before
being treated as final scientific settings.

#### What this detector is, and what it is not

The current nucleus detector is a **classical image-processing detector**, not a
neural network. Watershed is the final instance-separation algorithm: it divides a
foreground region around distance-map peaks so touching nuclei receive different
integer IDs. Watershed by itself does not learn weights from data.

This is an intentional baseline for the current repository because there are no
human-validated nucleus masks available for supervised training. Training a U-Net
on masks produced by this same classical detector would only teach the network to
imitate its errors; it would not provide independent evidence of improved nucleus
segmentation. The classical path is also deterministic, lightweight, and easy to
audit while the DAPI signal remains clear.

A future neural alternative would use a **Deep Watershed** design:

```text
DAPI image
    -> trained neural network
    -> predicted nucleus-interior and boundary/distance maps
    -> watershed post-processing
    -> nucleus instance labels
```

That upgrade requires either a compatible pretrained fluorescence-nucleus model or
a small set of manually validated nucleus instance masks. The downstream contract
would remain unchanged: background is `0`, every nucleus has a positive integer ID,
and the binary/proximity inputs are generated exactly as they are now. This allows
the detector to be replaced later without restructuring the astrocyte pipeline.

### Nucleus binary mask

Internal and optional external labels use the same 2D instance-image contract:

```text
0 = background
1, 2, 3, ... = individual nuclei
```

The model does not use the instance numbers directly. It receives:

```text
nucleus_mask = labels > 0
```

The source labels are still preserved because instance identity may become useful
for future cell assignment or per-nucleus analysis.

### Nucleus-proximity map

For every pixel, the Euclidean distance to the nearest nucleus pixel is clipped at
`max_nucleus_distance` and inverted:

```text
proximity = 1 - min(distance, max_distance) / max_distance
```

Consequently:

- nucleus pixels equal `1`;
- values decrease smoothly away from nuclei;
- pixels at or beyond the maximum distance equal `0`;
- an image with no positive nucleus pixels produces an all-zero proximity map.

The default maximum distance is `64` pixels.

### Patch extraction

The default patch settings are:

```text
patch size = 512 x 512
overlap = 64 pixels
stride = patch size - overlap = 448 pixels
```

Coordinates are deterministic and row-major. If the regular grid misses the far
edge, a final border-aligned patch is added. Images smaller than the configured
patch size produce one smaller patch rather than being padded.

Every patch retains its `image_id` and full-image `(y, x, height, width)`
coordinates. These coordinates allow evaluation and inference to reconstruct the
original image exactly.

The script `create_patches.py` writes only a coordinate index. It does not create
thousands of copied patch-image files. `AstrocyteDataset` performs lazy extraction
from full images and caches the most recently accessed full record.

## Splitting and grouped cross-validation

Never randomly split patches from one image across training and validation. Nearby
patches share pixels and biological content, which would create severe leakage and
overestimate performance.

Normal training uses the manifest's `split` column. Define splits before patch
generation, preferably at the highest meaningful biological level:

1. experiment or biological replicate;
2. plate or well;
3. complete image, if no higher grouping is available.

For a very small labeled dataset, grouped cross-validation is supported directly:

```yaml
cross_validation:
  enabled: true
  n_splits: 5
  validation_fold: 0
  group_column: image_id
  fold_column: fold
```

To group all fields from the same well, add a non-empty `well_id` column and use:

```yaml
group_column: well_id
```

The fold algorithm:

1. removes test rows and non-trainable annotation states;
2. assigns each complete group to one fold;
3. balances folds greedily by number of images;
4. uses the configured seed for deterministic tie-breaking;
5. verifies that train and validation groups do not overlap;
6. creates patch indices only after the grouped split exists.

Each run saves `cross_validation_assignments.csv` with the effective image/group,
fold, and split values. At least `n_splits` distinct non-empty groups are required.

## Baseline model and training

### U-Net

The implemented baseline is a compact 2D U-Net with:

- three encoder stages;
- a bottleneck;
- three skip-connected decoder stages;
- a final `1 x 1` convolution producing class logits;
- interpolation at skip connections so odd image dimensions are supported;
- configurable input channels, classes, and base channel width.

Default model configuration:

```yaml
model:
  architecture: unet
  input_channels: 3
  num_classes: 2
  base_channels: 32
```

`model_factory.py` is the single place that translates an architecture name into
a model. A SegFormer entry exists as an explicit placeholder, but it is not
implemented.

### Loss

The objective combines per-pixel categorical cross-entropy and soft Dice loss:

```text
total_loss = CE_weight * cross_entropy + Dice_weight * dice_loss
```

Both weights default to `1.0`. Background is excluded from the Dice term by
default, so overlap optimization focuses on the GFAP-positive class.

### Optimization and checkpoints

Training uses AdamW with configuration-controlled learning rate and weight decay.
After every epoch it records training/validation loss and foreground Dice.

The output directory contains:

```text
best.pt       checkpoint with the best validation Dice
last.pt       checkpoint from the most recent completed epoch
history.csv   per-epoch loss and Dice history
```

Each checkpoint stores model parameters, optimizer state, epoch, validation
metric, and the complete configuration. Early stopping ends training after the
configured number of validation epochs without Dice improvement.

Training automatically uses CUDA when available; otherwise it runs on CPU.

### Data augmentation

The current training augmentation is `RandomFlip`. It applies aligned horizontal
and vertical flips to both input and target. Validation and evaluation do not use
random augmentation.

## Evaluation and inference

### Full-image inference

Images are usually larger than GPU-friendly patches. Prediction therefore:

1. generates the same deterministic overlapping grid;
2. runs the model on each input patch;
3. applies softmax to obtain class probabilities;
4. averages probabilities where patches overlap;
5. applies `argmax` only after the full probability map is reconstructed.

Hard labels are never stitched directly. Averaging probabilities reduces border
discontinuities and preserves uncertainty information.

Standard prediction writes:

```text
outputs/predictions/probabilities/<image_id>.npy  # [classes, H, W]
outputs/predictions/masks/<image_id>.tiff         # [H, W] class IDs
outputs/predictions/overlays/<image_id>.png       # GFAP + prediction QC
```

### Metrics

The implemented metrics are:

- Dice score;
- intersection over union (IoU);
- precision;
- recall.

Values are calculated per class and as a macro average over foreground classes.
Empty-class cases use explicit finite rules rather than returning `NaN`.

Evaluation first reconstructs each complete image from its patches. Overlap pixels
are therefore counted once, rather than receiving extra weight because they occur
in multiple patches. The evaluation CSV contains one row per image and a final
`__aggregate__` row containing the mean across images.

## Iterative seed-to-correction workflow

The intended small-data loop is:

```text
manually corrected seed annotations
              |
              v
       initial U-Net training
              |
              v
 predictions on images with annotation_status=none
              |
              v
 pseudo labels + uncertainty-ranked patch queue
              |
              v
       manual correction/review
              |
              v
 retraining on seed + corrected + reviewed annotations
```

This loop can be repeated as annotation capacity allows. It does not require every
image to have a human astrocyte mask at the beginning.

## Installation

Python `>=3.11,<3.13` is supported by `pyproject.toml`.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it, then install the package and development dependencies:

```bash
pip install -e ".[dev]"
```

Confirm that the model, loss, optimizer, and backward pass work on a synthetic CPU
example:

```bash
python scripts/train.py --smoke-test --smoke-steps 25
```

Run the complete synthetic test suite:

```bash
pytest
```

No research images are required for either check.

## Complete command-line workflow

The commands below assume they are run from the repository root.

### 1. Add microscopy images

Place source `.tif` or `.tiff` files under `data/raw/`. Nested experiment
directories are supported.

### 2. Build the initial manifest

```bash
python scripts/build_manifest.py \
  --raw-dir data/raw \
  --output data/metadata/manifest.csv
```

The builder recursively discovers TIFF files, creates stable image IDs from paths,
and initializes new rows with `annotation_status=none`. It does not guess
experimental metadata. The next command can fill supported channel mappings.

Fill in experimental metadata needed for grouping and analysis. Supported RGB and
named OME channel mappings do not need to be entered image by image.

### 3. Automatically separate channels and detect nuclei

Run one command for every row in the manifest:

```powershell
.\.python311\python.exe scripts\prepare_dataset.py
```

For RGB TIFFs, the command uses Blue as DAPI and automatically chooses the stronger
of Red or Green as GFAP. Named OME channels use their metadata. It then detects
nuclei, updates `gfap_channel`, `dapi_channel`, and `cellpose_mask_path`, and saves:

```text
data/interim/channels/                  extracted GFAP and DAPI arrays
data/interim/nucleus_labels/            internal instance-label TIFFs
data/interim/nucleus_masks/             binary nucleus inputs
data/interim/nucleus_distance_maps/     proximity inputs
data/interim/qc/                        previews, montage, preparation report
```

Expected nucleus-mask contract:

```text
2D numeric array with the same H x W as the microscopy image
0 = background
positive integer = nucleus instance ID
```

Inspect `data/interim/qc/<image_id>_montage.png`. The montage includes GFAP, DAPI,
instance labels, binary nuclei, and the proximity map.

The nucleus result in this command comes from the classical detector described
above. It is automatic for every manifest image, but it is not a learned model.
Always inspect the QC montage when adding a new microscope, magnification, staining
condition, or image resolution.

### 4. Optional separate preprocessing commands

The unified command above replaces the need to run these manually. They remain
available when debugging one stage or when using externally generated labels.

Export already selected GFAP and DAPI arrays plus previews:

```bash
python scripts/extract_channels.py \
  --manifest data/metadata/manifest.csv \
  --output-dir data/interim/channels
```

Validate existing internal or external nucleus labels and regenerate masks,
proximity maps, previews, and QC montages:

```bash
python scripts/generate_nucleus_inputs.py \
  --manifest data/metadata/manifest.csv \
  --output-dir data/interim
```

Create a coordinate-only patch index for inspection:

```bash
python scripts/create_patches.py \
  --manifest data/metadata/manifest.csv \
  --output data/interim/patches/patch_index.csv \
  --patch-size 512 \
  --overlap 64
```

These commands are strongly useful for debugging alignment, but their outputs are
not prerequisites for `train.py`.

### 5. Import the first seed annotations

Create a pair table connecting manifest image IDs to the masks exported from your
annotation tool:

```csv
image_id,mask_path,annotation_status,annotation_source,annotator,review_status
img_001,exports/img_001_instances.tiff,seed,manual_cellpose_correction,AB,pending
img_003,exports/img_003_instances.tiff,seed,manual_cellpose_correction,AB,pending
```

Only `image_id` and `mask_path` are mandatory in the pair table; command-line
defaults supply metadata that is omitted.

Import the masks:

```bash
python scripts/import_existing_annotations.py \
  --manifest data/metadata/manifest.csv \
  --pairs-csv data/metadata/seed_annotation_pairs.csv \
  --output-dir data/annotations \
  --output-manifest data/metadata/manifest_seed.csv \
  --status seed \
  --source manual_cellpose_correction \
  --annotator AB
```

This command writes a **new manifest**. It also archives each source mask, creates
a binary target, and saves an alignment overlay. Inspect every overlay before
training.

### 6. Define the training and validation split

For standard training, set annotated rows to `train` or `val` in the active
manifest. Reserve biologically independent rows as `test` when possible.

For grouped cross-validation, enable the `cross_validation` block in
`configs/train_binary.yaml`. Use one image or well group per fold, never individual
patches.

### 7. Point the configuration to the active manifest

Edit:

```yaml
data:
  manifest_path: data/metadata/manifest_seed.csv
```

Review patch size, overlap, batch size, number of workers, and output directory
before starting a long run.

### 8. Train the initial model

Normal train/validation split:

```bash
python scripts/train.py --config configs/train_binary.yaml
```

One grouped validation fold:

```bash
python scripts/train.py \
  --config configs/train_binary.yaml \
  --fold 0
```

The `--fold` option automatically enables cross-validation and overrides
`cross_validation.validation_fold` for that run.

### 9. Evaluate the checkpoint

```bash
python scripts/evaluate.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --split val \
  --output outputs/metrics/validation.csv
```

When grouped cross-validation is enabled, evaluation recreates the same grouped
fold using the saved configuration values.

### 10. Predict complete images

```bash
python scripts/predict.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --split test \
  --output-dir outputs/predictions
```

This predicts every manifest row in the selected split, whether or not it has a
human annotation.

### 11. Generate pseudo labels for unlabeled images

The configuration's active manifest must include images in state `none` with valid
image, GFAP-channel, and nucleus-mask fields.

```bash
python scripts/generate_pseudo_labels.py \
  --config configs/train_binary.yaml \
  --checkpoint outputs/checkpoints/binary_baseline/best.pt \
  --output-dir outputs/pseudo_labels \
  --output-manifest data/metadata/manifest_with_pseudo.csv
```

The command predicts only `annotation_status=none` rows. It writes automatic
artifacts under `outputs/pseudo_labels/` and creates a new manifest whose matching
rows have `annotation_status=pseudo` and model provenance. The input manifest is
not silently overwritten.

### 12. Select uncertain patches for manual correction

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

The ranking score is mean normalized predictive entropy. Higher entropy means the
model is less certain. It is a prioritization heuristic, not a measure of
biological importance or annotation quality.

### 13. Import corrected masks

After manual correction, create another pair table and import it against the
pseudo-label manifest:

```bash
python scripts/import_existing_annotations.py \
  --manifest data/metadata/manifest_with_pseudo.csv \
  --pairs-csv data/metadata/corrected_annotation_pairs.csv \
  --output-dir data/annotations \
  --output-manifest data/metadata/manifest_corrected.csv \
  --status corrected \
  --source manual_pseudo_correction \
  --annotator AB \
  --overwrite
```

`--overwrite` allows the active derived target and lifecycle entry to advance.
The content-addressed copy of the original imported mask is still protected.

Update `data.manifest_path` to the corrected manifest and repeat training. The
default status filter now includes those corrected images automatically.

### 14. Extract preliminary features

```bash
python scripts/extract_features.py \
  --mask-dir outputs/predictions/masks \
  --output outputs/feature_tables/test_features.csv
```

The current command reports mask-derived field summaries in pixel units:

- positive pixel count;
- positive area fraction;
- connected-component count;
- skeleton length;
- branch-point count;
- endpoint count.

These are not validated single-cell measurements. Connected components are not
guaranteed to correspond to individual astrocytes, and cells or fields from the
same well must not be treated as independent biological replicates.

## Configuration reference

The normal binary experiment is defined in `configs/train_binary.yaml`.

```yaml
seed: 42

data:
  manifest_path: data/metadata/manifest.csv
  patch_size: 512
  overlap: 64
  num_workers: 4
  max_nucleus_distance: 64.0
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
  include_background_in_dice: false

cross_validation:
  enabled: false
  n_splits: 5
  validation_fold: 0
  group_column: image_id
  fold_column: fold

output:
  directory: outputs/checkpoints/binary_baseline
```

### Important settings

| Setting | Effect |
|---|---|
| `seed` | Controls deterministic initialization, augmentation randomness, and grouped-fold tie-breaking. |
| `data.manifest_path` | Selects the active data and annotation version for the run. |
| `data.patch_size` | Controls model crop size and inference tile size. |
| `data.overlap` | Controls shared pixels between neighboring patches. Must be smaller than patch size. |
| `data.num_workers` | Number of PyTorch data-loader worker processes. Use `0` when debugging. |
| `data.max_nucleus_distance` | Distance in pixels at which the proximity channel reaches zero. |
| `data.train_annotation_statuses` | Explicit lifecycle states allowed into training/evaluation. |
| `model.base_channels` | Width and approximate capacity/memory cost of the U-Net. |
| `training.batch_size` | Patches processed together; reduce it if GPU memory is insufficient. |
| `training.early_stopping_patience` | Validation epochs without improvement before stopping. |
| `cross_validation.group_column` | Unit that must remain intact across folds, such as `image_id` or `well_id`. |
| `output.directory` | Destination for checkpoints, history, and fold assignments. |

`configs/data.yaml` contains shared data defaults, and
`configs/annotation_workflow.yaml` records human/pseudo output locations and
selection defaults. `train_multiclass.yaml` is infrastructure for future work; it
does not mean that a validated multiclass model or annotation set currently
exists.

## Script reference

| Script | Reads | Writes | Purpose |
|---|---|---|---|
| `build_manifest.py` | Raw TIFF directory | Manifest CSV | Discovers images and creates conservative empty metadata rows. |
| `prepare_dataset.py` | Manifest and microscopy TIFFs | Channels, internal nucleus labels, model inputs, QC, updated manifest | Automatically prepares all images in one batch command. |
| `extract_channels.py` | Manifest and OME-TIFFs | GFAP/DAPI `.npy` files and previews | Makes channel selection easy to inspect. |
| `generate_nucleus_inputs.py` | Manifest, images, existing nucleus labels | Binary masks, proximity maps, previews, montages | Validates nucleus alignment and visualizes the model inputs. |
| `create_patches.py` | Manifest images | Patch-index CSV | Records deterministic patch coordinates without copying pixel arrays. |
| `import_existing_annotations.py` | Manifest and image-mask pair CSV | Archived originals, binary masks, QC overlays, new manifest | Adds seed/corrected/reviewed human targets non-destructively. |
| `train.py` | YAML config and annotated manifest | Checkpoints, history, optional fold assignments | Trains the configured baseline or runs the synthetic smoke test. |
| `evaluate.py` | Config, checkpoint, annotated split | Metrics CSV | Reconstructs complete images and reports segmentation metrics. |
| `predict.py` | Config, checkpoint, selected split | Probabilities, TIFF masks, overlays | Runs standard full-image prediction. |
| `generate_pseudo_labels.py` | Config, checkpoint, `none` rows | Automatic artifacts and new pseudo manifest | Creates model labels without mixing them with human data. |
| `select_unlabeled_patches.py` | Manifest and probability maps | Ranked annotation queue CSV | Finds high-entropy patches for manual work. |
| `extract_features.py` | Predicted mask directory | Feature-table CSV | Creates preliminary field-level morphology summaries. |

Every script documents all arguments:

```bash
python scripts/<script_name>.py --help
```

## How to read the code without getting lost

The easiest way to understand the project is to follow one data item through a
specific workflow instead of reading folders alphabetically.

### Training path

```text
configs/train_binary.yaml
    -> scripts/train.py
    -> io/manifest.py
    -> datasets/astrocyte_dataset.py
    -> preprocessing/{normalize,nuclei,distance_maps,patches}.py
    -> models/model_factory.py
    -> models/unet.py
    -> training/{losses,trainer,checkpoints}.py
```

Read these files in that order. `scripts/train.py` shows the orchestration;
`AstrocyteDataset` is where manifest rows become tensors; `UNet.forward` shows the
network; and `train_model` shows the optimization loop and saved outputs.

### Annotation-import path

```text
pair table CSV
    -> scripts/import_existing_annotations.py
    -> annotations/workflow.py
    -> visualization/overlays.py
    -> updated manifest CSV
```

The key function is `import_annotation_pair`. It contains the preservation,
validation, binarization, and overlay rules for one image-mask pair.

### Prediction path

```text
scripts/predict.py
    -> datasets.prepare_model_inputs
    -> inference/predict_full_image.py
    -> inference/predict_patch.py
    -> preprocessing.stitch_probability_patches
    -> probability, mask, and overlay files
```

The central idea is that full images are divided into patches only temporarily;
the output probabilities are reconstructed at the original resolution.

### Pseudo-label and correction path

```text
scripts/generate_pseudo_labels.py
    -> annotations/pseudo_labels.py
    -> outputs/pseudo_labels/
    -> scripts/select_unlabeled_patches.py
    -> annotations/selection.py
    -> manual correction
    -> scripts/import_existing_annotations.py
```

This path explains how automatic results remain separate until a human turns them
into a `corrected` or `reviewed` annotation.

## Tests

Tests use synthetic arrays, temporary TIFF files, and tiny neural networks. They
do not require the real microscopy dataset.

Current coverage includes:

- OME-TIFF axis handling, dtype, metadata, and named channels;
- normalization and nucleus-label validation;
- proximity-map range and edge cases;
- complete patch coverage and exact probability stitching;
- dataset lifecycle filtering, shapes, dtypes, and class ranges;
- U-Net output shape, including odd dimensions;
- Dice, IoU, precision, recall, and empty-class behavior;
- full-image evaluation after overlapping-patch reconstruction;
- non-destructive annotation import and source preservation;
- pseudo-label artifact separation;
- uncertainty selection and per-image limits;
- grouped image/well folds and leakage prevention.

Run:

```bash
pytest
```

The smoke test and unit tests establish software correctness. They do not establish
biological validity or model accuracy on the research dataset.

## Common errors and what they mean

### `Manifest contains no eligible annotated rows`

Check the requested `split`, `annotation_status`, and `annotation_path`. By default,
`none` and `pseudo` rows are intentionally excluded from supervised datasets.

### `Nucleus mask file does not exist`

Run `prepare_dataset.py`, or populate `cellpose_mask_path` with an external label
file. A nucleus mask is required even for prediction-only images.

### Annotation or nucleus shape does not match the image

The mask and OME image do not have identical spatial dimensions. Do not resize
blindly: first verify that they represent the same field, orientation, crop, and
resolution.

### Requested GFAP channel is missing

Inspect the actual channel names. RGB Red/Green/Blue and explicit GFAP/DAPI OME
names are selected automatically; other unnamed layouts require explicit manifest
values.

### Grouped cross-validation has too few groups

Reduce `n_splits` or choose a grouping column containing at least that many unique,
non-empty values. Do not solve this by splitting patches from the same image.

### Pseudo labels are not used during training

This is the expected default. Correct/review them first, or explicitly add
`pseudo` to `train_annotation_statuses` if an experiment intentionally uses
unreviewed model targets.

### GPU memory is insufficient

Reduce `training.batch_size`, then consider reducing `data.patch_size`. Changing
patch size also changes spatial context and should be recorded as an experimental
decision.

## Current limitations

- Internal nucleus detection uses classical Otsu, distance peaks, and watershed; it
  is automatic but is not a neural network and must be validated on every new image
  condition.
- Deep Watershed is not currently implemented. Adding it responsibly requires a
  compatible pretrained model or manually validated nucleus instance masks; the
  existing label-file interface is designed so such a model can replace the current
  detector without changing downstream training.
- External Cellpose labels remain supported, but Cellpose execution is not embedded.
- The supported scientific target is binary GFAP-positive structure segmentation.
- The multiclass configuration is future scaffolding, not a completed experiment.
- SegFormer is an explicit `NotImplementedError` placeholder.
- No trained checkpoint or research dataset is bundled with the repository.
- Prediction processes patches sequentially rather than in inference batches.
- Checkpoints contain optimizer state, but there is no resume-training CLI yet.
- Physical-unit conversion is not applied in the current feature-extraction script.
- Connected components are not reliable individual-astrocyte identities.
- Nucleus assignment, skeleton, branch, and endpoint outputs remain exploratory.
- Notebooks are currently placeholders for project-specific exploration.

## Development principles

- Preserve raw microscopy, optional Cellpose outputs, and original annotation exports.
- Keep human-reviewed targets separate from automatic predictions.
- Use the manifest as the source of truth for paths, metadata, and lifecycle state.
- Define image/well splits before creating patches.
- Keep reusable logic in `src/astroseg` and command orchestration in `scripts/`.
- Store the configuration and fold assignments needed to reproduce each run.
- Fail clearly on missing files, channels, invalid labels, or misaligned dimensions.
- Prefer a small, reproducible baseline over an unverified complex model.
