from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred, eps: float = 1e-12) -> float:
    """MAPE with safe handling of zeros in ``y_true``."""

    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def coefficient_of_determination(y_true, y_pred) -> float:
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else np.nan


def huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    error = y_true - y_pred
    small = np.abs(error) <= delta
    return float(np.mean(np.where(small, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))))


def quantile_loss(y_true, y_pred, q: float = 0.5) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def evaluate_predictions(y_true, y_pred, phase_names=None, split_name: str = "test") -> pd.DataFrame:
    """Compute per-phase metrics for arrays shaped ``(N, rows, cols, phases)``."""

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    n_phases = y_true.shape[-1]
    if phase_names is None:
        phase_names = [f"phase_{i + 1}" for i in range(n_phases)]

    rows = []
    for i, phase in enumerate(phase_names):
        yt = y_true[..., i].ravel()
        yp = y_pred[..., i].ravel()
        rows.append(
            {
                "split": split_name,
                "phase": phase,
                "mse": mean_squared_error(yt, yp),
                "rmse": mean_squared_error(yt, yp) ** 0.5,
                "mae": mean_absolute_error(yt, yp),
                "r2": r2_score(yt, yp),
                "mape": mean_absolute_percentage_error(yt, yp),
                "cd": coefficient_of_determination(yt, yp),
                "huber_loss": huber_loss(yt, yp),
                "quantile_loss": quantile_loss(yt, yp),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(model, x_train, y_train, x_test, y_test, phase_names=None) -> pd.DataFrame:
    """Run model predictions and return train/test metrics in one table."""

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    return pd.concat(
        [
            evaluate_predictions(y_train, y_train_pred, phase_names, split_name="train"),
            evaluate_predictions(y_test, y_test_pred, phase_names, split_name="test"),
        ],
        ignore_index=True,
    )
