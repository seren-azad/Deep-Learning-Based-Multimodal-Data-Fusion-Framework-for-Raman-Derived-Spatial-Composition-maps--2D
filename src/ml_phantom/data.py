from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import DatasetConfig


@dataclass(frozen=True)
class PairRecord:
    """One image/CSV pair found on disk."""

    image_path: Path
    csv_path: Path
    sample_folder: str
    augmentation_folder: str


def discover_pairs(config: DatasetConfig, strict: bool = False) -> list[PairRecord]:
    """Find image/CSV pairs under the configured sample and augmentation folders.

    Pairing rules are intentionally permissive:
    1. first try a CSV with the same stem as the image;
    2. then try the notebook convention where image names beginning with
       ``P_sub_`` correspond to CSV names beginning with ``A_sub_``;
    3. if neither is found, optionally raise an error or skip the image.
    """

    base = Path(config.base_path)
    records: list[PairRecord] = []
    missing: list[Path] = []

    for main_folder in config.main_folders:
        for sub_folder in config.sub_folders:
            folder = base / main_folder / sub_folder
            if not folder.exists():
                if strict:
                    raise FileNotFoundError(f"Missing folder: {folder}")
                continue

            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() not in config.image_extensions:
                    continue
                csv_path = _matching_csv_path(image_path)
                if csv_path is None:
                    missing.append(image_path)
                    continue
                records.append(
                    PairRecord(
                        image_path=image_path,
                        csv_path=csv_path,
                        sample_folder=main_folder,
                        augmentation_folder=sub_folder,
                    )
                )

    if strict and missing:
        msg = "Missing CSV files for images:\n" + "\n".join(str(p) for p in missing[:20])
        if len(missing) > 20:
            msg += f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(msg)

    return records


def _matching_csv_path(image_path: Path) -> Path | None:
    same_stem = image_path.with_suffix(".csv")
    if same_stem.exists():
        return same_stem

    alt_name = image_path.stem.replace("P_sub_", "A_sub_") + ".csv"
    alt_path = image_path.with_name(alt_name)
    if alt_path.exists():
        return alt_path

    return None


def load_image(image_path: str | Path, image_size: tuple[int, int], normalize: bool = True) -> np.ndarray:
    """Load an image as a grayscale array with shape ``(height, width, 1)``."""

    with Image.open(image_path) as img:
        img = img.convert("L")
        if img.size != (image_size[1], image_size[0]):
            img = img.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)

    if normalize:
        arr = arr / 255.0

    return arr[..., np.newaxis]


def load_label_grid(
    csv_path: str | Path,
    grid_shape: tuple[int, int] = (6, 6),
    phase_names: Iterable[str] = ("B", "C1", "C2", "C3", "C4", "C5"),
    image_size: tuple[int, int] = (150, 150),
    normalize_labels: bool = False,
) -> np.ndarray:
    """Load one CSV file and return a label grid with shape ``(rows, cols, phases)``.

    If the CSV contains ``x`` and ``y`` columns, they are interpreted as spatial
    coordinates and used to place rows into the grid. Otherwise the CSV is
    reshaped in its current row order.
    """

    phase_names = tuple(phase_names)
    df = pd.read_csv(csv_path)
    missing = [name for name in phase_names if name not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required phase columns: {missing}")

    rows, cols = grid_shape
    n_expected = rows * cols
    values = df.loc[:, phase_names].to_numpy(dtype=np.float32)

    if len(values) != n_expected:
        raise ValueError(
            f"{csv_path} contains {len(values)} rows, but {n_expected} rows are required "
            f"for grid_shape={grid_shape}."
        )

    if {"x", "y"}.issubset(df.columns):
        label_grid = np.zeros((rows, cols, len(phase_names)), dtype=np.float32)
        cell_h = image_size[0] / rows
        cell_w = image_size[1] / cols
        for _, row in df.iterrows():
            cell_x = int(float(row["x"]) // cell_w)
            cell_y = int(float(row["y"]) // cell_h)
            cell_x = min(max(cell_x, 0), cols - 1)
            cell_y = min(max(cell_y, 0), rows - 1)
            label_grid[cell_y, cell_x, :] = row.loc[list(phase_names)].to_numpy(dtype=np.float32)
    else:
        label_grid = values.reshape(rows, cols, len(phase_names))

    if normalize_labels:
        sums = label_grid.sum(axis=-1, keepdims=True)
        label_grid = np.divide(label_grid, sums, out=np.zeros_like(label_grid), where=sums != 0)

    return label_grid


def load_dataset(config: DatasetConfig, strict: bool = False) -> tuple[np.ndarray, np.ndarray, list[PairRecord]]:
    """Load all image/CSV pairs into NumPy arrays."""

    records = discover_pairs(config, strict=strict)
    if not records:
        raise FileNotFoundError(f"No image/CSV pairs found under {config.base_path}")

    images = []
    labels = []
    for record in records:
        images.append(load_image(record.image_path, config.image_size, config.normalize_images))
        labels.append(
            load_label_grid(
                record.csv_path,
                grid_shape=config.grid_shape,
                phase_names=config.phase_names,
                image_size=config.image_size,
                normalize_labels=config.normalize_labels,
            )
        )

    return np.stack(images, axis=0), np.stack(labels, axis=0), records


def save_dataset_npz(
    output_path: str | Path,
    images: np.ndarray,
    labels: np.ndarray,
    records: list[PairRecord],
) -> None:
    """Save loaded arrays and source paths to a compressed ``.npz`` file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        image_paths=np.array([str(r.image_path) for r in records]),
        csv_paths=np.array([str(r.csv_path) for r in records]),
        sample_folders=np.array([r.sample_folder for r in records]),
        augmentation_folders=np.array([r.augmentation_folder for r in records]),
    )


def load_dataset_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["images"], data["labels"]


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train/test split matching the original notebook workflow."""

    return train_test_split(images, labels, test_size=test_size, random_state=random_state)
