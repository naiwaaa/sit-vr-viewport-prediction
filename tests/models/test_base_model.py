from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import FakeModel
    from viewport_prediction.types import NDArrayFloat


def test_build_fake_model(fake_model: FakeModel) -> None:
    result = fake_model

    assert len(result.model.layers) == 1
    assert result.model.layers[0].units == 1


def test_plot_fake_model_to_temp_file(fake_model: FakeModel) -> None:
    result = fake_model.plot_model(out_file=None)

    assert result.ndim == 3


def test_plot_fake_model_to_actual_file(tmp_path: Path, fake_model: FakeModel) -> None:
    out_file = tmp_path / "fake_model.png"

    result = fake_model.plot_model(out_file)

    assert result.ndim == 3
    assert out_file.is_file()
    assert out_file.exists()


def test_save_fake_model(tmp_path: Path, fake_model: FakeModel) -> None:
    out_dir = tmp_path / "saved_model"

    fake_model.save(out_dir)  # act

    assert out_dir.is_dir()
    assert out_dir.exists()
    assert len(list(out_dir.iterdir())) > 0


def test_print_fake_model_summary(fake_model: FakeModel) -> None:
    fake_model.summary()  # act


def test_fake_model_prediction(fake_model: FakeModel) -> None:
    data_x: NDArrayFloat = np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=float)
    data_y: NDArrayFloat = np.asarray([[3.0], [5.0]], dtype=float)

    result = fake_model.predict([data_x])

    assert result.shape == (2, 2)
    assert np.allclose(result, data_y * [np.pi, 2 * np.pi])
