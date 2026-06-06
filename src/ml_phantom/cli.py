from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import DatasetConfig
from .data import load_dataset, save_dataset_npz, load_dataset_npz, split_dataset
from .metrics import evaluate_model
from .training.train import train_from_npz


def preprocess_main(argv=None):
    parser = argparse.ArgumentParser(description="Load image/CSV pairs and save a processed NPZ dataset.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--base-path", type=str, default=None, help="Base data path.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path.")
    parser.add_argument("--start-sample", type=int, default=1)
    parser.add_argument("--end-sample", type=int, default=9)
    parser.add_argument("--exclude-brightness", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if args.config:
        config = DatasetConfig.from_json(args.config)
    else:
        if args.base_path is None:
            parser.error("--base-path is required when --config is not provided")
        config = DatasetConfig(
            base_path=args.base_path,
            main_folders=[f"S{i}" for i in range(args.start_sample, args.end_sample + 1)],
        )

    if args.exclude_brightness:
        config = DatasetConfig(
            **{**config.__dict__, "sub_folders": [s for s in config.sub_folders if s != "patches_Brightness"]}
        )

    images, labels, records = load_dataset(config, strict=args.strict)
    save_dataset_npz(args.output, images, labels, records)
    print(f"Saved {len(records)} image/CSV pairs to {args.output}")
    print(f"images shape: {images.shape}")
    print(f"labels shape: {labels.shape}")


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train one of the phantom fraction models.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", choices=["model1", "model2", "model3"], default="model3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    _, _, test_loss = train_from_npz(
        dataset_path=args.dataset,
        model_name=args.model,
        output_model=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Saved model to {args.output}")
    print(f"Test metrics: {test_loss}")


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained Keras models on a processed NPZ dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    images, labels = load_dataset_npz(args.dataset)
    x_train, x_test, y_train, y_test = split_dataset(
        images, labels, test_size=args.test_size, random_state=args.random_state
    )

    all_metrics = []
    for model_path in args.models:
        model = tf.keras.models.load_model(model_path)
        df = evaluate_model(model, x_train, y_train, x_test, y_test)
        df.insert(0, "model", Path(model_path).stem)
        all_metrics.append(df)

    out = pd.concat(all_metrics, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved metrics to {args.output}")
