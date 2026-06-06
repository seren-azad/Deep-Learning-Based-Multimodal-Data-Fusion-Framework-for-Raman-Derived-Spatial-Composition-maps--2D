from __future__ import annotations

from pathlib import Path
import random
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from ml_phantom.data import load_dataset_npz, split_dataset
from ml_phantom.models.architectures import compile_model, get_model_builder


def set_random_seed(seed: int = 42, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def default_callbacks(output_model: str | Path, patience: int = 5, min_delta: float = 1e-3):
    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=max(2, patience // 2),
            min_delta=min_delta,
            min_lr=1e-6,
        ),
        ModelCheckpoint(output_model, monitor="val_loss", save_best_only=True),
        CSVLogger(output_model.with_suffix(".training_log.csv")),
    ]


def train_from_npz(
    dataset_path: str | Path,
    model_name: str,
    output_model: str | Path,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
):
    set_random_seed(random_state)
    images, labels = load_dataset_npz(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(
        images, labels, test_size=test_size, random_state=random_state
    )

    builder = get_model_builder(model_name)
    model = builder(input_shape=x_train.shape[1:], num_phases=y_train.shape[-1], grid_cells=y_train.shape[1])
    model = compile_model(model, use_decay=(model_name.lower() == "model3"))

    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=default_callbacks(output_model),
    )
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    return model, history, test_loss
