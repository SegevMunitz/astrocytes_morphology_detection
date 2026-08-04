# Astrocyte Segmentation Pipeline

A research-oriented image-analysis project for separating complete individual
GFAP-positive astrocytes in multichannel microscopy TIFF and OME-TIFF images.

The final task is **nucleus-guided astrocyte instance segmentation**. Every cell
receives a distinct positive ID, and every cell pixel is classified as nucleus,
soma, or process. The most important ownership output assigns each process to the
nucleus—and therefore the astrocyte—that owns it.

The repository retains the existing binary semantic pipeline as an intermediate
bootstrap rather than replacing it:

```text
0 = background
1 = GFAP-positive astrocyte structure (merged field)
```

That binary mask is useful for initial foreground proposals and annotation, but it
is not the final scientific output. The final output contract is:

```text
instance_labels: 0 = background, 1..N = individual complete astrocytes
compartments:    0 = background, 1 = nucleus, 2 = soma, 3 = process
ownership:       cell instance ID -> one nucleus instance ID
```

## Scientific motivation

The DAPI channel provides a strong signal for nucleus detection, but a nucleus mask
is not a complete astrocyte mask. Thin GFAP-positive processes can extend far
from the nucleus, overlap with neighboring structures, and are often not captured
well by an instance-segmentation method focused on compact objects.

This project therefore uses nuclei as both spatial context and cell-identity
anchors, rather than treating a nucleus mask as the complete cell. For every pixel,
the model combines:

1. GFAP fluorescence, which contains the structure to segment.
2. A binary nucleus mask, derived from internal or externally supplied instance labels.
3. A smooth nucleus-proximity map, which tells the model how close the pixel is
   to the nearest detected nucleus.

The instance U-Net shares one encoder/decoder and learns three aligned outputs:

1. semantic compartments: background, nucleus, soma, and process;
2. cell boundaries: evidence that touching structures belong to different cells;
3. ownership offsets: a two-dimensional vector from every cell pixel toward the
   nucleus that owns it.

The third head is essential for long or crossing processes. A nearest-nucleus rule
cannot know ownership when a process passes closer to another cell. Learned offsets
can use image appearance and annotated examples to predict the intended owner.

## The project in one diagram

```text
microscopy TIFF
    |-- GFAP channel --------------------> normalized appearance -----------+
    `-- DAPI channel --> automatic nucleus instances                        |
                              |-- binary nucleus mask ----------------------|
                              `-- nucleus proximity ------------------------+
                                                                            |
complete-cell instance annotation + optional compartments ------------------+
                                                                            v
                                                     shared 2D instance U-Net
                                      +----------------+----------+----------+
                                      |                |          |
                                      v                v          v
                               compartments       boundaries  ownership offsets
                                      |                |          |
                                      +----------------+----------+
                                                       v
                                  one ID per complete nucleus+soma+process cell
```

The manifest is the index connecting all these files. One manifest row describes
one microscopy image, its channel names, its nucleus labels, its annotation state,
and its train/validation/test assignment.

## The masks must not be confused

There are four related but distinct label files:

| File | Manifest column | Content | Role |
|---|---|---|---|
| Nucleus instances | `cellpose_mask_path` (legacy name) | `0` background, positive nucleus IDs | Input/QC and identity anchors for every image |
| Binary GFAP mask | `annotation_path` | `0/1` merged astrocyte foreground | Existing binary baseline and bootstrap |
| Complete-cell instances | `instance_annotation_path` | `0` background, positive astrocyte IDs | Authoritative process-ownership supervision |
| Compartments | `compartment_annotation_path` | `0` background, `1` nucleus, `2` soma, `3` process | Optional explicit compartment supervision |

The nucleus instance mask is needed for **every image passed through the model**,
including unlabeled images used only for prediction. Complete-cell annotations are
needed only for instance training and held-out evaluation.

This means that it is normal to have:

- nucleus masks for all usable images;
- complete-cell annotations for only a small subset;
- binary or automatic proposals for more images;
- no human astrocyte annotation yet for most images.

`prepare_dataset.py` detects nuclei internally from DAPI and writes the resulting
instance-label path to `cellpose_mask_path`. The column keeps its historical name
for compatibility with the existing dataset and training code. External Cellpose
labels remain supported as an optional alternative.

### Files required by each operation

| Operation | TIFF/GFAP | Nucleus IDs | Binary mask | Complete-cell IDs |
|---|---:|---:|---:|---:|
| Preparation/QC | Yes | Generated | No | No |
| Binary bootstrap/training | Yes | Yes | Proposal/target | No |
| Instance training | Yes | Yes | Derived/optional | **Yes** |
| Trained instance prediction | Yes | Yes | No | No |
| Instance evaluation | No | No | No | Prediction + human truth |
| Complete-cell import | Yes | Yes | Derived automatically | Imported input |

The DAPI channel is useful for inspection and QC, but it is not one of the three
current model-input channels.

## What one instance-model sample contains

For one complete image, the input array has this channel-first shape:

```text
[C, H, W] = [3, H, W]
```

| Channel | Construction | Range | Purpose |
|---|---|---:|---|
| `0`: GFAP | Named OME channel, percentile-normalized | `[0, 1]` | Main appearance signal |
| `1`: nucleus mask | `cellpose_labels > 0` | `{0, 1}` | Exact detected-nucleus locations |
| `2`: nucleus proximity | Inverted, clipped Euclidean distance | `[0, 1]` | Smooth distance-to-nucleus context |

`AstrocyteInstanceDataset` returns one image patch and all aligned supervision:

```python
{
    "image": torch.Tensor,           # [3, H, W], float32
    "targets": {
        "semantic": torch.Tensor,    # [H, W], classes 0..3
        "boundary": torch.Tensor,    # [H, W], classes 0/1
        "offsets": torch.Tensor,     # [2, H, W], normalized dy/dx to owner nucleus
        "offset_mask": torch.Tensor, # [H, W], regression support
    },
    "instances": torch.Tensor,       # [H, W], original cell IDs
    "image_id": str,
    "coordinates": PatchCoordinates,
}
```

The model returns `semantic_logits [B,4,H,W]`, `boundary_logits [B,2,H,W]`, and
`offsets [B,2,H,W]`. Training constructs targets lazily from the paths in the manifest. The
intermediate files generated by `prepare_dataset.py`, `extract_channels.py`,
`generate_nucleus_inputs.py`, and `create_patches.py` are useful for inspection,
QC, and debugging. The old `AstrocyteDataset` and binary U-Net remain available
for the foreground baseline.

## Repository structure

```text
astrocytes_morphology_detection/
|
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- data.yaml
|   |-- train_binary.yaml
|   |-- train_instances.yaml
|   |-- train_multiclass.yaml        legacy semantic experiment scaffold
|   `-- annotation_workflow.yaml
|
|-- data/
|   |-- raw/                         original microscopy images
|   |-- interim/                     optional preprocessing and QC outputs
|   |-- annotations/
|   |   |-- originals/               archived human-exported masks
|   |   |-- binary/                  derived binary training targets
|   |   |-- compartment_originals/   preserved optional component labels
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
|-- notebooks/                       guided executable pipeline and QC workflow
`-- outputs/
    |-- checkpoints/                 trained model states and histories
    |-- predictions/                 binary semantic model outputs
    |-- astrocyte_instances/         watershed bootstrap cell proposals
    |-- instance_predictions/        trained ownership-model outputs
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
| `preprocessing/gfap_detection.py` | `detect_gfap_bootstrap_mask` | Creates an initial automatic GFAP pseudo mask with connected weak/strong intensity thresholds; it is not treated as human ground truth. |
| `preprocessing/normalize.py` | `percentile_normalize` | Scales GFAP intensities to `[0, 1]` for model input. The result is not intended for biological intensity comparisons. |
| `preprocessing/nuclei.py` | `validate_nucleus_labels`, `labels_to_binary_mask` | Checks exact image alignment and valid instance labels, then converts all positive nucleus IDs to foreground. |
| `preprocessing/distance_maps.py` | `create_nucleus_proximity_map` | Converts the binary nucleus mask into a bounded distance-derived context channel. |
| `preprocessing/patches.py` | `PatchCoordinates`, `generate_patch_coordinates`, `stitch_probability_patches` | Creates deterministic, border-covering patch coordinates and reconstructs complete probability maps by averaging overlaps. |
| `datasets/astrocyte_dataset.py` | `prepare_model_inputs`, `AstrocyteDataset` | Builds the shared three inputs and serves the retained binary semantic baseline. |
| `datasets/instance_dataset.py` | `AstrocyteInstanceDataset` | Validates complete-cell IDs, builds compartment/boundary/ownership targets, and extracts aligned patches. |
| `models/unet.py` | `UNet`, `NucleusGuidedInstanceUNet` | Implements the retained semantic U-Net and final three-head instance model. |
| `preprocessing/instance_targets.py` | `build_astrocyte_instance_targets` | Maps each complete cell to one nucleus and creates semantic, boundary, and offset supervision. |
| `training/losses.py` | `CrossEntropyDiceLoss`, `NucleusGuidedInstanceLoss` | Optimizes compartment overlap, separation boundaries, and process-to-nucleus offsets jointly. |
| `training/instance_trainer.py` | `train_instance_model` | Trains all heads, monitors validation Dice, checkpoints, and provides a synthetic smoke test. |
| `training/trainer.py` | `train_model`, `run_overfit_smoke_test` | Runs AdamW optimization, validation, early stopping, checkpointing, history export, and a synthetic end-to-end diagnostic. |
| `training/cross_validation.py` | `assign_grouped_folds`, `load_grouped_fold_manifests` | Assigns entire images or wells to deterministic folds and checks that no group leaks across train and validation. |
| `training/metrics.py` | `metrics_from_predictions`, `metrics_from_probability_patches` | Computes finite Dice, IoU, precision, and recall after reconstructing complete images. |
| `inference/predict_instances.py` | `predict_instance_full_image` | Stitches semantic, boundary, and continuous ownership heads at full resolution. |
| `postprocessing/astrocyte_instances.py` | `separate_astrocyte_instances` | Converts learned ownership votes and nucleus anchors into complete individual cell IDs. |
| `training/instance_metrics.py` | `instance_segmentation_metrics`, `process_ownership_accuracy` | Measures cell detection/overlap and whether process pixels retain the correct owner. |
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
data/annotations/compartment_originals/<image_id>/<status>_<content_hash>.<ext>
data/annotations/qc/<image_id>_<status>_annotation_overlay.png
```

Automatic pseudo labels are stored separately:

```text
outputs/pseudo_labels/probabilities/<image_id>.npy
outputs/pseudo_labels/masks/<image_id>.tiff
outputs/pseudo_labels/overlays/<image_id>.png
```

Automatic complete-cell outputs are also separate:

```text
outputs/astrocyte_instances/labels/           watershed bootstrap proposals
outputs/instance_predictions/raw_heads/       learned probabilities and offsets
outputs/instance_predictions/labels/          final automatic cell IDs
outputs/instance_predictions/compartments/    automatic nucleus/soma/process IDs
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
| `instance_annotation_path` | Preserved human complete-cell ID mask | Required for final-model training |
| `compartment_annotation_path` | Optional preserved 0/1/2/3 component mask | When compartments were annotated explicitly |
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
image_id,path,gfap_channel,dapi_channel,cellpose_mask_path,annotation_path,instance_annotation_path,compartment_annotation_path,annotation_status,annotation_source,annotator,review_status,split,well_id
img_001,data/raw/img_001.tif,GFAP,DAPI,data/interim/nucleus_labels/img_001.tiff,data/annotations/binary/img_001_seed_binary.tiff,data/annotations/originals/img_001/seed_abc.tiff,data/annotations/compartment_originals/img_001/seed_def.tiff,seed,manual_complete_cell_correction,AB,pending,train,A01
img_002,data/raw/img_002.tif,GFAP,DAPI,data/interim/nucleus_labels/img_002.tiff,,,,none,,,,test,A02
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

`import_existing_annotations.py` checks binary annotations. For final-task data,
`import_instance_annotations.py` additionally checks that:

- the image ID exists exactly once in the manifest;
- the OME-TIFF and mask paths exist;
- the selected GFAP channel exists;
- the mask is a two-dimensional numeric array;
- image and mask dimensions are identical;
- mask labels are finite, non-negative, and integer-valued;
- existing derived artifacts are not overwritten unless requested.
- every positive astrocyte ID overlaps exactly one detected nucleus;
- one nucleus is not assigned to several astrocyte IDs;
- optional compartment values are only `0, 1, 2, 3` and stay inside their cell.

Dimension equality is a necessary alignment check, but it cannot prove biological
registration if the exported mask lacks complete spatial metadata. Always inspect
the generated GFAP overlay.

Instance-valued astrocyte masks are converted using `mask > 0` only to preserve
compatibility with the binary pipeline. The original instance IDs are archived
unchanged and recorded in `instance_annotation_path`; those IDs are the actual
supervision for complete-cell and process-ownership training.

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

## Models and training

### Final nucleus-guided instance U-Net

`configs/train_instances.yaml` defines the final-task model. A shared U-Net feature
map feeds three heads:

```text
semantic head -> background / nucleus / soma / process
boundary head -> interior / between-cell boundary
offset head   -> normalized dy, dx from cell pixel to owning nucleus
```

The total objective is:

```text
L = semantic_weight * L_semantic
  + boundary_weight * L_boundary
  + offset_weight * SmoothL1(owner_offset)
```

The offset term is evaluated only on annotated cell pixels. Horizontal and
vertical augmentation also changes the matching offset sign, so vector targets
remain physically correct after flipping.

At inference, offset endpoints vote for detected nucleus markers. Boundary-aware
watershed fills only votes whose endpoint is unreliable. Before a trained instance
checkpoint exists, the same postprocessor can run in `watershed_bootstrap` mode;
that output is a pseudo proposal, not validated process ownership.

### Retained binary U-Net

The existing compact 2D U-Net remains useful for generating or learning a merged
GFAP foreground mask. It has:

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

### Binary loss

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

The binary path uses `RandomFlip`. The instance path uses `RandomInstanceFlip`,
which also negates `dx` after a horizontal flip and `dy` after a vertical flip.
Validation and evaluation never use random augmentation.

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

Semantic metrics are:

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

Complete-cell evaluation additionally reports object precision/recall/F1 at a
specified IoU, mean matched IoU, panoptic quality, and process ownership accuracy.
Cells are matched one-to-one with Hungarian assignment, preventing one merged cell
from receiving credit for several annotated astrocytes.

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

To run the guided notebooks, install the notebook dependencies as well:

```bash
pip install -e ".[dev,notebooks]"
python -m jupyter lab
```

Select the kernel belonging to this environment. The notebooks also add the
repository's `src/` directory explicitly, so they work when Jupyter starts from
either the repository root or `notebooks/`.

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

## Guided notebook workflow

The notebooks call the same tested modules and scripts used by the command-line
workflow; they do not maintain a second implementation of the pipeline.

For routine operation, open `00_run_pipeline.ipynb`, review its **User settings**
cell, and choose **Run All**. It runs automatic preparation and GFAP pseudo
segmentation, summarizes outputs, and exposes guarded switches for later training,
evaluation, and prediction. It does not require running notebooks 01–04 first.

Notebooks 01–04 are the detailed learning, inspection, parameter-comparison, and
troubleshooting path. Run those in order when learning the project or investigating
one stage:

| Notebook | Purpose | Safe stopping point |
|---|---|---|
| `00_run_pipeline.ipynb` | Operational Run-All interface for manifest handling, preprocessing, nucleus detection, GFAP pseudo segmentation, QC, and optional trained-model stages. | Complete automatic result for the current data, or later model outputs when enabled. |
| `01_inspect_tiff_channels.ipynb` | Discover/load the manifest, inspect TIFF metadata and channels, run automatic preparation, and display its report and QC montage. | Prepared channels, nucleus labels, proximity maps, and manifest. |
| `02_visualize_nucleus_masks.ipynb` | Validate label alignment, visualize nucleus boundaries, inspect instance sizes, and optionally compare detector parameters without overwriting files. | Accepted nucleus QC. |
| `03_create_initial_annotations.ipynb` | Generate automatic GFAP bootstrap pseudo labels, inspect masks/confidence/overlays, compare thresholds, and optionally import a human-corrected mask. | Automatic `pseudo` mask, or an imported `corrected` target. |
| `04_evaluate_baseline.ipynb` | Audit training readiness, inspect the U-Net contract, run the synthetic smoke test, and optionally train, evaluate, and predict. | Smoke-test success until enough human-validated images exist. |

Only edit cells labeled **User settings**. Expensive or scientifically gated steps
use explicit Boolean switches such as `RUN_TRAINING`; executing all cells with their
defaults is safe for the current single test image. Automatic predictions remain
`pseudo` and are not converted into trusted training targets by a notebook.

The second notebook replaced the historical
`02_visualize_cellpose_masks.ipynb` name. The active manifest column remains
`cellpose_mask_path` for compatibility, but the notebook supports both internally
detected and externally supplied nucleus instance labels.

## Complete-cell workflow: the final task

The following is the shortest operational path from prepared images to true
individual-astrocyte training.

### 1. Prepare channels and nuclei automatically

```powershell
.\.python311\python.exe scripts\prepare_dataset.py
.\.python311\python.exe scripts\generate_bootstrap_pseudo_labels.py --overwrite
```

### 2. Generate a bootstrap cell proposal for correction

```powershell
.\.python311\python.exe scripts\generate_astrocyte_instances.py --overwrite
```

This writes cell IDs, compartments, per-cell measurements, and two QC overlays to
`outputs/astrocyte_instances/`. It uses nucleus-seeded watershed because no trained
ownership checkpoint exists yet. Use it to start annotation; do not treat its
process-to-cell assignments as ground truth.

### 3. Correct complete cells, not only foreground

In the annotation tool, every astrocyte must have a consistent positive ID across
its nucleus, soma, and all visible processes. Different astrocytes must have
different IDs. Export a 2D integer TIFF without resizing or interpolation.

An optional second TIFF may explicitly mark compartments:

```text
0 background, 1 nucleus, 2 soma, 3 process
```

Create a pair table:

```csv
image_id,instance_mask_path,compartment_mask_path,annotation_status,annotator,review_status
7d_453,exports/7d_453_cells.tiff,exports/7d_453_compartments.tiff,seed,AB,pending
```

Import it non-destructively:

```powershell
.\.python311\python.exe scripts\import_instance_annotations.py `
  --manifest data/metadata/manifest.csv `
  --pairs-csv data/metadata/instance_pairs.csv `
  --output-manifest data/metadata/manifest_instances.csv `
  --annotator AB
```

The importer derives the binary mask for compatibility, archives the exact source
IDs, validates dimensions and cell-to-nucleus mapping, and generates cell-ID and
compartment overlays.

### 4. Train with grouped folds

Point `configs/train_instances.yaml:data.manifest_path` at the new manifest. With
several images from the same well, add a non-empty `well_id` column and set
`cross_validation.group_column: well_id`. Otherwise leave `image_id`. Then run:

```powershell
# Engineering test that needs no real annotations
.\.python311\python.exe scripts\train_instances.py --smoke-test

# Real fold 0
.\.python311\python.exe scripts\train_instances.py `
  --config configs/train_instances.yaml `
  --fold 0
```

Folds are assigned before patches are generated, so patches from one image/well
cannot leak between training and validation. With fewer than the configured number
of independent groups, reduce `n_splits`; do not split patches from one image to
artificially create validation data.

### 5. Predict individual cells with learned process ownership

```powershell
.\.python311\python.exe scripts\predict_astrocyte_instances.py `
  --config configs/train_instances.yaml `
  --checkpoint outputs/checkpoints/astrocyte_instances/best.pt `
  --split test
```

The command saves raw model heads in addition to final labels and never populates
human annotation columns. Correct selected automatic cell masks, import them as
`corrected`, and retrain.

### 6. Evaluate held-out complete cells

```powershell
.\.python311\python.exe scripts\evaluate_astrocyte_instances.py `
  --manifest outputs/instance_predictions/manifest.csv
```

Process ownership accuracy requires explicit compartment annotations. Object
precision/recall/F1 and panoptic quality require human complete-cell IDs.

## Binary bootstrap and reference workflow

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

### Optional bootstrap: generate an automatic GFAP proposal

When no trained astrocyte checkpoint or manual seed mask exists yet, generate an
initial proposal directly from the separated GFAP channel:

```powershell
.\.python311\python.exe scripts\generate_bootstrap_pseudo_labels.py
```

The detector percentile-normalizes GFAP, smooths noise, uses Otsu to identify
strong signal, and retains weaker pixels only when they are connected to strong
signal. This hysteresis step recovers many dim processes without accepting every
faint background pixel. Tiny components are removed and small enclosed holes are
filled.

The command writes:

```text
outputs/pseudo_labels/probabilities/<image_id>.npy
outputs/pseudo_labels/masks/<image_id>.tiff
outputs/pseudo_labels/overlays/<image_id>.png
outputs/pseudo_labels/bootstrap_report.csv
outputs/pseudo_labels/manifest.csv
```

The source manifest is not modified. Rows in the generated manifest use
`annotation_status=pseudo`, `review_status=pending`, and
`annotation_source=heuristic:gfap_hysteresis_otsu`. Pseudo rows remain excluded
from supervised training by default.

This result is automatic but it is not a validated annotation and it is not a
trained neural-network prediction. It can be used immediately for visual testing,
as a mask to correct, or as an explicitly configured experimental pseudo target.
For scientifically supervised training, correct and import at least a small subset
as `corrected` or `reviewed`.

For new staining or acquisition conditions, inspect the overlays and tune
`--low-threshold-ratio`, `--high-threshold-scale`, or `--min-component-area` if
needed. Use `--overwrite` only when intentionally regenerating automatic files.

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

The final complete-cell experiment is defined in `configs/train_instances.yaml`.

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
  architecture: nucleus_guided_instance_unet
  input_channels: 3
  num_classes: 4
  base_channels: 32

training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 15

loss:
  semantic_weight: 1.0
  boundary_weight: 1.0
  offset_weight: 2.0

cross_validation:
  enabled: true
  n_splits: 5
  validation_fold: 0
  group_column: image_id
  fold_column: instance_fold

output:
  directory: outputs/checkpoints/astrocyte_instances
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
| `data.soma_radius` | Radius used only when explicit compartment labels are absent. |
| `data.offset_scale` | Pixel scale used to normalize nucleus-directed ownership vectors. |
| `data.train_annotation_statuses` | Explicit lifecycle states allowed into training/evaluation. |
| `model.base_channels` | Width and approximate capacity/memory cost of the shared U-Net. |
| `loss.offset_weight` | Importance of correct cell ownership along soma/process pixels. |
| `training.batch_size` | Patches processed together; reduce it if GPU memory is insufficient. |
| `training.early_stopping_patience` | Validation epochs without improvement before stopping. |
| `cross_validation.group_column` | Unit that must remain intact across folds, such as `image_id` or `well_id`. |
| `output.directory` | Destination for checkpoints, history, and fold assignments. |

`configs/train_binary.yaml` retains the merged-foreground baseline. `configs/data.yaml` contains shared data defaults, and
`configs/annotation_workflow.yaml` records human/pseudo output locations and
selection defaults. `train_multiclass.yaml` is older semantic scaffolding; use
`train_instances.yaml` for distinct complete cells.

## Script reference

| Script | Reads | Writes | Purpose |
|---|---|---|---|
| `build_manifest.py` | Raw TIFF directory | Manifest CSV | Discovers images and creates conservative empty metadata rows. |
| `prepare_dataset.py` | Manifest and microscopy TIFFs | Channels, internal nucleus labels, model inputs, QC, updated manifest | Automatically prepares all images in one batch command. |
| `extract_channels.py` | Manifest and OME-TIFFs | GFAP/DAPI `.npy` files and previews | Makes channel selection easy to inspect. |
| `generate_nucleus_inputs.py` | Manifest, images, existing nucleus labels | Binary masks, proximity maps, previews, montages | Validates nucleus alignment and visualizes the model inputs. |
| `generate_bootstrap_pseudo_labels.py` | Manifest and GFAP channels | Heuristic probabilities, masks, overlays, report, separate pseudo manifest | Bootstraps proposals before a trained astrocyte checkpoint exists. |
| `generate_astrocyte_instances.py` | Pseudo GFAP + nucleus IDs | Watershed cell proposals, compartments, overlays, measurements | Creates correction-ready bootstrap instances without claiming learned ownership. |
| `create_patches.py` | Manifest images | Patch-index CSV | Records deterministic patch coordinates without copying pixel arrays. |
| `import_existing_annotations.py` | Manifest and image-mask pair CSV | Archived originals, binary masks, QC overlays, new manifest | Adds seed/corrected/reviewed human targets non-destructively. |
| `import_instance_annotations.py` | Manifest, complete-cell IDs, optional components | Preserved instance truth, derived binary mask, QC, new manifest | Validates one cell-to-one nucleus ownership for final-model training. |
| `train.py` | YAML config and annotated manifest | Checkpoints, history, optional fold assignments | Trains the configured baseline or runs the synthetic smoke test. |
| `train_instances.py` | Instance config and complete-cell annotations | Multi-head checkpoints, history, grouped assignments | Trains compartment, boundary, and ownership heads jointly. |
| `evaluate.py` | Config, checkpoint, annotated split | Metrics CSV | Reconstructs complete images and reports segmentation metrics. |
| `predict.py` | Config, checkpoint, selected split | Probabilities, TIFF masks, overlays | Runs standard full-image prediction. |
| `predict_astrocyte_instances.py` | Instance checkpoint and selected split | Raw heads, individual cells, compartments, overlays, measurements | Runs learned complete-cell separation. |
| `evaluate_astrocyte_instances.py` | Human and predicted instance paths | Object/ownership metric tables | Evaluates individual cells instead of only foreground pixels. |
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
scripts/generate_bootstrap_pseudo_labels.py  (before the first checkpoint)
                or
scripts/generate_pseudo_labels.py            (after model training)
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
- automatic GFAP bootstrap hysteresis and cleanup;
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
- Bootstrap GFAP masks are intensity-based heuristic proposals, not validated
  ground truth or neural-network predictions. They remain `pseudo` and are excluded
  from supervised training unless explicitly corrected or experimentally enabled.
- External Cellpose labels remain supported, but Cellpose execution is not embedded.
- The final architecture is implemented, but no trained instance checkpoint or
  human complete-cell research dataset is bundled. Current `7d_453` cell separation
  is therefore a watershed bootstrap, not validated learned process ownership.
- In a two-dimensional image, truly overlapping processes can be intrinsically
  ambiguous. The model can learn annotation conventions and image cues, but cannot
  recover information absent from the acquisition; z-stacks or additional markers
  may be needed for difficult crossings.
- SegFormer is an explicit `NotImplementedError` placeholder.
- No trained checkpoint or research dataset is bundled with the repository.
- Prediction processes patches sequentially rather than in inference batches.
- Checkpoints contain optimizer state, but there is no resume-training CLI yet.
- Physical-unit conversion is not applied in the current feature-extraction script.
- Binary connected components are not reliable individual-astrocyte identities.
  Use instance IDs from the final model and validate them on held-out corrections.
- Per-cell skeleton, branch, and endpoint measurements remain exploratory until the
  complete-cell masks and compartment definitions are biologically reviewed.
- Notebook defaults demonstrate preparation and bootstrap separation; real learned
  process ownership remains gated on human complete-cell annotations and independent folds.

## Development principles

- Preserve raw microscopy, optional Cellpose outputs, and original annotation exports.
- Keep human-reviewed targets separate from automatic predictions.
- Use the manifest as the source of truth for paths, metadata, and lifecycle state.
- Define image/well splits before creating patches.
- Keep reusable logic in `src/astroseg` and command orchestration in `scripts/`.
- Store the configuration and fold assignments needed to reproduce each run.
- Fail clearly on missing files, channels, invalid labels, or misaligned dimensions.
- Prefer a small, reproducible baseline over an unverified complex model.
