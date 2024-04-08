from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import PositiveInt, DirectoryPath, PositiveFloat
from pydantic.generics import GenericModel

from viewport_prediction.config.base_config import BaseConfig


class BaseModelConfig(BaseConfig):
    past_window_size: int
    future_window_size: int


ModelConfigT = TypeVar("ModelConfigT", bound="BaseModelConfig")


class DataConfig(BaseConfig):
    data_dir: DirectoryPath

    train_video_indices: list[PositiveInt]
    val_video_indices: list[PositiveInt] | None
    test_video_indices: list[PositiveInt]


class TrainingConfig(BaseConfig):
    loss_func: Literal["mse", "orthodromic_distance"]
    optimizer: Literal["adam"]

    max_epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat


class WandbConfig(BaseConfig):
    project: str
    entity: str
    tags: list[str]
    resume: bool


class MiscConfig(BaseConfig):
    seed: PositiveInt | None


class ExperimentConfig(BaseConfig, GenericModel, Generic[ModelConfigT]):
    data: DataConfig
    model: ModelConfigT
    training: TrainingConfig
    wandb: WandbConfig
    misc: MiscConfig | None
