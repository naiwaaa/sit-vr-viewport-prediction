# pylint: disable=redefined-outer-name
from typing import TypeAlias

from pathlib import Path

import pytest

from tensorflow import keras as tfk

from viewport_prediction.models import BaseModel
from viewport_prediction.config.experiment_config import (
    BaseModelConfig,
    ExperimentConfig,
)


class FakeModelConfig(BaseModelConfig):
    hidden_size: int


class FakeModel(BaseModel[FakeModelConfig]):
    """A fake model used for testing."""

    Config: TypeAlias = FakeModelConfig

    def build(self) -> None:
        """Build the fake model's architecture."""
        self.model = tfk.Sequential(
            [
                tfk.layers.Dense(
                    units=self.config.model.hidden_size,
                    input_shape=(None, 2),
                    kernel_initializer="ones",
                    bias_initializer="zeros",
                ),
            ],
        )


# ------------------------
# fixtures return file/dir
# ------------------------


def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def lena_image_file() -> Path:
    return fixtures_dir() / "lena.png"


@pytest.fixture()
def realshort_video_file() -> Path:
    return fixtures_dir() / "realshort.mp4"


@pytest.fixture()
def minimal_yml_config_file() -> Path:
    return fixtures_dir() / "config_files" / "minimal.yml"


@pytest.fixture()
def minimal_toml_config_file() -> Path:
    return fixtures_dir() / "config_files" / "minimal.toml"


@pytest.fixture()
def minimal_json_config_file() -> Path:
    return fixtures_dir() / "config_files" / "minimal.json"


@pytest.fixture()
def fake_model_config_file() -> Path:
    return fixtures_dir() / "config_files" / "fake_model.toml"


@pytest.fixture()
def ieee_letter_2020_config_file() -> Path:
    return fixtures_dir() / "config_files" / "ieee_letter_2020_model.toml"


@pytest.fixture()
def raw_head_orientation_log_file() -> Path:
    return fixtures_dir() / "raw_head_orientation_log.json"


@pytest.fixture()
def subtitle_file() -> Path:
    return fixtures_dir() / "subtitle.srt"


@pytest.fixture()
def has_subtitle_session_dir() -> Path:
    return fixtures_dir() / "has_subtitle" / "video_05_user_99"


# ------------------------
# fixtures create instance
# ------------------------


@pytest.fixture()
def fake_model(fake_model_config_file: Path) -> FakeModel:
    config = ExperimentConfig[FakeModelConfig].read_from_file(fake_model_config_file)
    model = FakeModel(config)
    model.build()
    return model
