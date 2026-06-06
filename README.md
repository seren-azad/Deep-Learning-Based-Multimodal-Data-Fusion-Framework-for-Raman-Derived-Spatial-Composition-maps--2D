## Dataset

The dataset used in this project is available on Zenodo:

**Dataset:** [Deep Learning-Based Multimodal Data Fusion Framework for Raman Derived Spatial Composition maps 2D](https://doi.org/10.5281/zenodo.20572145)

Please download and extract the dataset before running the scripts.



# ML models

Python package for predicting phase area fractions from 2D phantom patch images.

This repository was refactored from a Jupyter notebook workflow. The original notebook is kept in `notebooks/ML_fusion_2D_GIT_original.ipynb`, while the reusable code is organized under `src/ml_phantom/`.

## Scientific task

The model receives normalized grayscale patch images and predicts the area fractions of six phases in a `6 × 6` grid:

```text
Input image:      (150, 150, 1)
Target label:     (6, 6, 6)
Phase channels:   B, C1, C2, C3, C4, C5
```

The target values are continuous fractions, so the task is treated as multi-output regression rather than classification.

## Repository structure

```text
configs/                  Configuration files
docs/                     Method documentation and migration notes
examples/                 Small examples for local execution
notebooks/                Original and cleaned notebooks
scripts/                  Command-line scripts
src/ml_phantom/           Reusable Python package
tests/                    Lightweight unit tests
data/                     Ignored local data folder
outputs/                  Ignored outputs for models and figures
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -e .[dev]
```

TensorFlow can be heavy and GPU-dependent. Install the version appropriate for your machine if the default dependency does not match your setup.

## Data layout

The loader expects this type of folder structure:

```text
C:/Users/Mahdieh/NK/ML/phantom/
├── S1/
│   ├── patches/
│   ├── patches_Brightness/
│   ├── patches_flipped/
│   ├── patches_Noise/
│   └── patches_rotated/
├── S2/
...
└── S9/
```

Each subfolder should contain image/CSV pairs. The image can be `.jpg`, `.jpeg`, or `.png`. The CSV must contain the six phase columns:

```text
B, C1, C2, C3, C4, C5
```

If `x` and `y` columns are present, they are used to place the phase fractions in the `6 × 6` grid. Otherwise, the rows are reshaped in file order.

## Preprocess data

```bash
ml-phantom-preprocess \
  --base-path "C:/Users/Mahdieh/NK/ML/phantom" \
  --output "data/processed/phantom_dataset.npz" \
  --start-sample 1 \
  --end-sample 9
```

This creates an `.npz` file containing:

```text
images:      (N, 150, 150, 1)
labels:      (N, 6, 6, 6)
image_paths: source image paths
csv_paths:   source CSV paths
```

## Train a model

```bash
ml-phantom-train \
  --dataset data/processed/phantom_dataset.npz \
  --model model3 \
  --output outputs/models/model3.keras \
  --epochs 100
```

Available model names:

```text
model1  basic encoder with global pooling
model2  residual encoder with grid attention
model3  squeeze-and-excitation residual encoder with grid attention
```

## Evaluate trained models

```bash
ml-phantom-evaluate \
  --dataset data/processed/phantom_dataset.npz \
  --models outputs/models/model1.keras outputs/models/model2.keras outputs/models/model3.keras \
  --output outputs/evaluation_metrics.csv
```

## Run tests

```bash
pytest
```

## Data policy

Raw data, processed `.npz` files, trained models, and generated figures are ignored by Git. Keep large or unpublished datasets outside the repository, or publish them separately with appropriate permissions.
