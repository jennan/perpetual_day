from pathlib import Path
from typing import Any, Literal

import pydantic as pdt

import src.models


class Config(pdt.BaseModel):
    start_date: str
    end_date: str
    interval: str
    bbox: tuple[float, float, float, float]
    cachedir: Path
    resultsdir: Path
    ckpt_path: Path | None = None  # used to resume training
    n_jobs: int = 1
    seed: int = 42
    test_size: float | int
    datamodule_params: dict[str, Any]
    model_class: Literal["CNN", "Unet", "DiffUNet2D"]
    model_params: dict[str, Any]
    trainer_params: dict[str, Any]

    @property
    def model(self):
        if self.model_class == "CNN":
            return src.models.CNN
        elif self.model_class == "Unet":
            return src.models.Unet
        elif self.model_class == "DiffUNet2D":
            return src.models.DiffUNet2D
        else:
            raise ValueError(f"Unknown model class {self.model_class}")
