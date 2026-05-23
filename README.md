# wsi-mil

A Multiple Instance Learning (MIL) library for training, inference, and visualization on Whole Slide Images (WSI).

# Installation

```bash
uv add git+https://github.com/ymd000/wsi-mil.git
uv sync
source .venv/bin/activate
```

# Preprocessing

Before training, preprocess whole-slide images using the following toolbox:

[`wsi_toolbox`](https://github.com/technoplasm/wsi-toolbox/)

This toolbox is used for whole-slide image patch extraction and HDF5 generation.

# Usage

## Aggregation (TITAN)

**Python**
```python
import wsi_mil as mil

dataset = mil.WSIDataset(
    data_dir="data/h5",
    encoder_name="conch15_768",
)

cmd = mil.TITANAggregateCommand(
    model_name_or_path="MahmoodLab/TITAN",
    config=mil.TITANConfig(patch_size_lv0=512),
)
results = cmd(dataset)
```

**CLI**
```bash
mil aggregate --method titan \
  --data-dir /path/to/hdf5 --encoder conch15_768

# with config file
mil aggregate --config configs/aggregate_titan.yaml

# config file with CLI override
mil aggregate --config configs/aggregate_titan.yaml --data-dir /new/path
```

**`patch_size_lv0` reference**

| Extraction condition | patch_size_lv0 |
|----------------------|----------------|
| level 1 (downsample=2.0) / 256px | 512 |
| level 1 (downsample=2.0) / 512px | 1024 |

**Flow**
```
HDF5
├── {encoder}/features    (N, D)
└── {encoder}/coordinates (N, 2)
        ↓ TITANAggregateCommand
HDF5
└── {encoder}/slide_embedding/titan/embedding (D,)
```

---

## Aggregation (Simple)

**Python**
```python
import wsi_mil as mil

dataset = mil.WSIDataset(
    data_dir="data/h5",
    encoder_name="uni2",
)

cmd = mil.SimpleAggregateCommand(
    config=mil.SimpleAggregateConfig(method="mean_pooling"),
)
results = cmd(dataset)
```

**CLI**
```bash
mil aggregate --method mean_pooling \
  --data-dir /path/to/hdf5 --encoder uni2

mil aggregate --method nearest_cosine \
  --data-dir /path/to/hdf5 --encoder uni2
```

---

## Aggregation (ABMIL)

**Python**
```python
import wsi_mil as mil

dataset = mil.WSIDataset(
    data_dir="data/h5",
    encoder_name="uni2",
)

cmd = mil.ABMILAggregateCommand(
    model_class=mil.ABMIL,
    config=mil.ABMILAggregateConfig(
        method="abmil",
        output_dir="./outputs",
        version="latest",
        checkpoint_name="best",
        use_val_fold=True,
        model_kwargs={"input_dim": 1536, "num_classes": 2},
    ),
)
results = cmd(dataset)
```

**CLI**
```bash
mil aggregate --method abmil \
  --data-dir /path/to/hdf5 --encoder uni2 \
  --output-dir ./outputs --version latest

# with config file
mil aggregate --config configs/aggregate_abmil.yaml
```

**Flow**
```
HDF5
└── {encoder}/features (N, D)
        ↓ TrainCommand        → outputs/version_X/fold_N/checkpoints/best.ckpt
        ↓ ABMILAggregateCommand    method="abmil" or "abmil_top"
HDF5
└── {encoder}/slide_embedding/abmil/
    ├── embedding      (D,)
    ├── attention      (N,)
    └── probabilities  (num_classes,)
```

---

## Aggregate CLI Options

`--config` is optional. CLI arguments take precedence over YAML values (priority: CLI > YAML > defaults).

| Flag | Methods | Default | Description |
|------|---------|---------|-------------|
| `--method` | all | `mean_pooling` | `mean_pooling` / `nearest_cosine` / `nearest_euclidean` / `titan` / `abmil` / `abmil_top` |
| `--data-dir` | all | — | Directory containing HDF5 files |
| `--encoder` | all | — | Encoder name (e.g. `conch15_768`) |
| `--device` | all | `auto` | `auto` / `cuda` / `cpu` |
| `--normalize` | simple | `false` | L2-normalize the slide embedding |
| `--model` | titan | `MahmoodLab/TITAN` | Model path or HuggingFace ID |
| `--patch-size` | titan | `512` | Level-0 patch size in pixels |
| `--output-dir` | abmil | `./outputs` | Training output directory |
| `--version` | abmil | `latest` | Version to load (`latest` or int) |
| `--checkpoint` | abmil | `best` | Checkpoint name (`best` or `last`) |

---

## Training (ABMIL)

**Python**
```python
import wsi_mil as mil

dataset = mil.WSIDataset(
    data_dir="data/h5",
    encoder_name="uni2",
    csv_path="data/labels.csv",
)

cmd = mil.TrainCommand(
    model_class=mil.ABMIL,
    config=mil.TrainConfig(
        model_kwargs={"input_dim": 1536, "num_classes": 2},
        num_fold=5,
        max_epochs=50,
        output_dir="./outputs",
    ),
)
results = cmd(dataset)
```

**Config** (`configs/train.yaml`)
```yaml
output_dir: ./outputs
num_fold: 5
max_epochs: 50
lr: 1.0e-3
model_kwargs:
  input_dim: 1536
  num_classes: 2

dataset:
  data_dir: /path/to/hdf5
  encoder: uni2
  csv_path: /path/to/labels.csv
```

**Train Output**
```
outputs/
└── version_0/
    ├── config.yaml
    ├── fold_indices.csv
    ├── fold_0/
    │   ├── checkpoints/
    │   │   ├── best.ckpt
    │   │   └── last.ckpt
    │   └── logs/
    │       └── metrics.csv
    └── fold_1/ ...
```

---

## Evaluation

**Python**
```python
import wsi_mil as mil

results = mil.EvaluateCommand.load_embeddings(
    data_dir="data/h5",
    method_name="abmil",
    csv_path="data/labels.csv",
    encoder_name="uni2",
)

cmd = mil.EvaluateCommand(
    config=mil.EvaluateConfig(
        output_dir="./eval",
        class_names={0: "Benign", 1: "Malignant"},
        plot_umap=True,
        plot_confusion_matrix=True,
        show_misclassified=True,
    )
)
metrics = cmd(results)
```

**Config** (`configs/evaluate.yaml`)
```yaml
method_name: abmil
output_dir: ./eval_outputs
positive_class: 1
average: macro
class_names:
  0: Benign
  1: Malignant
plot_umap: true
plot_confusion_matrix: true
show_misclassified: true

dataset:
  data_dir: /path/to/hdf5
  encoder: uni2
  csv_path: /path/to/labels.csv
  # subtype_csv_path: /path/to/subtype.csv
  # subtype_col: subtype
```

**CLI**
```bash
mil evaluate --config configs/evaluate.yaml
```

# HDF5 Structure

Follows the [`wsi_toolbox`](https://github.com/technoplasm/wsi-toolbox/) specification.

```
sample.h5
├── {encoder}/
│   ├── features                      # patch embeddings (N, D)
│   └── coordinates                   # level 0 coordinates (N, 2) (required for TITAN)
└── {encoder}/slide_embedding/
    ├── abmil/
    │   ├── embedding                 # (D,)
    │   ├── attention                 # (N,)
    │   └── probabilities             # (num_classes,)
    ├── abmil_top/
    │   ├── embedding                 # (D,)
    │   ├── attention                 # (N,)
    │   └── selected_index            # attr
    ├── mean_pooling/
    │   └── embedding                 # (D,)
    ├── nearest_cosine/
    │   ├── embedding                 # (D,)
    │   └── selected_index            # attr
    ├── nearest_euclidean/
    │   ├── embedding                 # (D,)
    │   └── selected_index            # attr
    └── titan/
        └── embedding                 # (D,)
```

---

## Encoder Dimensions

| encoder | dim |
|---------|-----|
| `uni` | 1024 |
| `uni2` | 1536 |
| `gigapath` | 1536 |
| `virchow2` | 2560 |
| `conch15_768` | 768 |

## Dependencies

- **transformers**: Required for TITAN (`TITANAggregateCommand`)
