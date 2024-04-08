from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


def round_to_half(array: NDArrayFloat) -> NDArrayFloat:
    """Round to nearest 0.5.

    Args:
        array (NDArrayFloat): Input data

    Returns:
        NDArrayFloat: An array containing the rounded values.

    Examples:
        >>> round_to_half([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        array([0. , 0. , 0. , 0.5, 0.5, 0.5])

        >>> round_to_half([0.6, 0.7, 0.8, 0.9, 1.0])
        array([0.5, 0.5, 1. , 1. , 1. ])
    """
    rounded_array: NDArrayFloat = (
        np.round(np.asarray(array, dtype=float) * 2, decimals=0) / 2
    )
    return rounded_array
