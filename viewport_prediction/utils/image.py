from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage


if TYPE_CHECKING:
    from typing import Literal

    from pathlib import Path

    from viewport_prediction.types import NDArrayUInt8


def imread(
    file: Path,
    color_mode: Literal["grayscale", "rgb"],
    target_size: tuple[int, int] | None = None,
) -> NDArrayUInt8:
    """Read an image from a file and return it as a numpy array.

    Args:
        file (Path): The image file to read.
        color_mode (Literal["grayscale", "rgb"]): The desired image format.
        target_size (Tuple[int, int], optional): A tuple `(target_height, target_width)`.

    Returns:
        NDArrayUInt8: The image data.
    """
    img = PILImage.open(file)

    if color_mode == "grayscale" and img.mode != "L":
        img = img.convert("L")

    if color_mode == "rgb" and img.mode != "RGB":
        img = img.convert("RGB")  # pragma: no cover

    if target_size is not None:
        img = img.resize(
            (target_size[1], target_size[0]),
            resample=PILImage.NEAREST,
        )

    return np.asarray(img, dtype=np.uint8)


def imwrite(
    out_file: Path,
    data: NDArrayUInt8,
    color_mode: Literal["grayscale", "rgb"],
) -> None:
    """Write an image to the specified file."""
    mode: Literal["L", "RGB"] = "L" if color_mode == "grayscale" else "RGB"
    img = PILImage.fromarray(data, mode)
    img.save(out_file)
