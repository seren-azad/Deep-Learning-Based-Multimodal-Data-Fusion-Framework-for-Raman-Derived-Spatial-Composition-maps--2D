# Migration notes from notebook to package

The original notebook contained one continuous workflow: preprocessing, exploratory data analysis, train/test split, three model definitions, model training, prediction visualization, and final model comparison.

This repository separates those responsibilities:

| Notebook section | Refactored location |
|---|---|
| Data loading | `src/ml_phantom/data.py` |
| Dataset configuration | `src/ml_phantom/config.py` |
| EDA and distribution checks | `src/ml_phantom/eda.py` |
| Model 1, 2, 3 definitions | `src/ml_phantom/models/architectures.py` |
| Training callbacks and training loop | `src/ml_phantom/training/train.py` |
| Prediction overlays and loss plots | `src/ml_phantom/visualization.py` |
| Per-phase model metrics | `src/ml_phantom/metrics.py` |
| Command-line execution | `src/ml_phantom/cli.py` |

The original notebook is retained for provenance. The package code is the version intended for reuse and GitHub publication.
