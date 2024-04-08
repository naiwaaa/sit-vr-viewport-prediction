from __future__ import annotations

from typing import TYPE_CHECKING

from viewport_prediction.models.base_model import BaseModel
from viewport_prediction.models.ieee_letter_2020 import (
    IEEELetter2020Model,
    IEEELetter2020ModelConfig,
)


if TYPE_CHECKING:
    from viewport_prediction.config.experiment_config import BaseModelConfig


ALL_MODELS: dict[str, type[BaseModel[BaseModelConfig]]] = {
    "ieee2020": IEEELetter2020Model,  # type: ignore
}

__all__ = [
    "ALL_MODELS",
    "BaseModel",
    "IEEELetter2020ModelConfig",
    "IEEELetter2020Model",
]
