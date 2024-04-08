from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from viewport_prediction.utils import image


if TYPE_CHECKING:
    from pathlib import Path

    from viewport_prediction.types import NDArrayUInt8


def test_imread_rgb(lena_image_file: Path) -> None:
    result = image.imread(lena_image_file, color_mode="rgb")

    assert result.shape == (512, 512, 3)


def test_imread_grayscale(lena_image_file: Path) -> None:
    result = image.imread(lena_image_file, color_mode="grayscale")

    assert result.shape == (512, 512)


def test_imread_rescale(lena_image_file: Path) -> None:
    target_width, target_height = 12, 9

    result = image.imread(
        lena_image_file,
        color_mode="grayscale",
        target_size=(target_height, target_width),
    )

    assert result.shape == (target_height, target_width)


def test_imwrite(tmp_path: Path) -> None:
    img_data: NDArrayUInt8 = np.random.randint(
        low=0,
        high=255,
        size=(10, 10, 3),
    ).astype(np.uint8)
    out_file = tmp_path / "random.jpg"
    image.imwrite(out_file, img_data, color_mode="rgb")

    result = image.imread(out_file, color_mode="rgb")

    assert result.shape == (10, 10, 3)
