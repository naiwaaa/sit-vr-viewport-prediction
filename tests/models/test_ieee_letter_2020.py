from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import numpy as np

from viewport_prediction.data import HFDataset
from viewport_prediction.config import ExperimentConfig
from viewport_prediction.entities import Session
from viewport_prediction.models.ieee_letter_2020 import (
    IEEELetter2020Model,
    IEEELetter2020ModelConfig,
)


if TYPE_CHECKING:
    from pathlib import Path

    from viewport_prediction.types import NDArrayFloat, NDArrayUInt8


def build_model(
    config_file: Path,
) -> tuple[IEEELetter2020Model, ExperimentConfig[IEEELetter2020ModelConfig]]:
    config = ExperimentConfig[IEEELetter2020ModelConfig].read_from_file(config_file)

    model = IEEELetter2020Model(config)
    model.build()

    return model, config


def test_build_ieee_letter_2020_model(ieee_letter_2020_config_file: Path) -> None:
    build_model(ieee_letter_2020_config_file)  # act


@pytest.mark.slow()
def test_fit_ieee_letter_2020_model(
    has_subtitle_session_dir: Path,
    ieee_letter_2020_config_file: Path,
) -> None:
    model, config = build_model(ieee_letter_2020_config_file)
    dataset = HFDataset(
        [Session(has_subtitle_session_dir)],
        past_window_size=config.model.past_window_size,
        future_window_size=config.model.future_window_size,
    )

    model.fit(  # act
        dataset.loader(batch_size=config.training.batch_size),
        val_dataloader=None,
        callbacks=[],
    )


def test_ieee_letter_2020_model_predict(ieee_letter_2020_config_file: Path) -> None:
    model, _ = build_model(ieee_letter_2020_config_file)
    #
    input_head_orientation: NDArrayFloat = np.random.rand(1, 5, 2)  # noqa: SIM903
    input_video_frame: NDArrayUInt8 = np.random.randint(
        low=0,
        high=255,
        size=(1, 5, 224, 224, 3),
    ).astype(dtype=np.uint8)

    result = model.predict([input_head_orientation, input_video_frame])

    assert result.shape == (1, 5, 2)


@pytest.mark.slow()
def test_evaluate_ieee_letter_2020_model(
    has_subtitle_session_dir: Path,
    ieee_letter_2020_config_file: Path,
) -> None:
    model, config = build_model(ieee_letter_2020_config_file)
    dataset = HFDataset(
        [Session(has_subtitle_session_dir)],
        past_window_size=config.model.past_window_size,
        future_window_size=config.model.future_window_size,
    )

    model.evaluate(dataset.loader(batch_size=config.training.batch_size))  # act
