import numpy as np
from ml_phantom.metrics import evaluate_predictions


def test_evaluate_predictions_shape():
    y_true = np.ones((2, 6, 6, 6), dtype=float)
    y_pred = np.ones((2, 6, 6, 6), dtype=float) * 0.9
    df = evaluate_predictions(y_true, y_pred, phase_names=["B", "C1", "C2", "C3", "C4", "C5"])
    assert len(df) == 6
    assert set(["phase", "mse", "mae", "r2"]).issubset(df.columns)
