from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for image/CSV pair loading."""

    base_path: str | Path
    main_folders: list[str] = field(default_factory=lambda: [f"S{i}" for i in range(1, 10)])
    sub_folders: list[str] = field(
        default_factory=lambda: [
            "patches",
            "patches_Brightness",
            "patches_flipped",
            "patches_Noise",
            "patches_rotated",
        ]
    )
    image_size: tuple[int, int] = (150, 150)
    grid_shape: tuple[int, int] = (6, 6)
    phase_names: tuple[str, ...] = ("B", "C1", "C2", "C3", "C4", "C5")
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    normalize_images: bool = True
    normalize_labels: bool = False

    @property
    def n_phases(self) -> int:
        return len(self.phase_names)

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "image_size" in data:
            data["image_size"] = tuple(data["image_size"])
        if "grid_shape" in data:
            data["grid_shape"] = tuple(data["grid_shape"])
        if "phase_names" in data:
            data["phase_names"] = tuple(data["phase_names"])
        if "image_extensions" in data:
            data["image_extensions"] = tuple(data["image_extensions"])
        return cls(**data)
