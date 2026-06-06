from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml_phantom.config import DatasetConfig
from ml_phantom.data import discover_pairs, load_dataset, load_label_grid


def _write_pair(folder: Path, stem: str = "sample"):
    folder.mkdir(parents=True, exist_ok=True)
    image = (np.random.rand(150, 150) * 255).astype("uint8")
    Image.fromarray(image).save(folder / f"{stem}.jpg")
    rows = []
    for y in [12.5, 37.5, 62.5, 87.5, 112.5, 137.5]:
        for x in [12.5, 37.5, 62.5, 87.5, 112.5, 137.5]:
            rows.append({"x": x, "y": y, "B": 1, "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0})
    pd.DataFrame(rows).to_csv(folder / f"{stem}.csv", index=False)


def test_load_label_grid(tmp_path):
    folder = tmp_path / "S1" / "patches"
    _write_pair(folder)
    label = load_label_grid(folder / "sample.csv")
    assert label.shape == (6, 6, 6)
    assert np.all(label[..., 0] == 1)


def test_load_dataset(tmp_path):
    folder = tmp_path / "S1" / "patches"
    _write_pair(folder)
    config = DatasetConfig(base_path=tmp_path, main_folders=["S1"], sub_folders=["patches"])
    records = discover_pairs(config)
    images, labels, records = load_dataset(config)
    assert len(records) == 1
    assert images.shape == (1, 150, 150, 1)
    assert labels.shape == (1, 6, 6, 6)
    assert images.min() >= 0
    assert images.max() <= 1
