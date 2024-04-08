from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


def rmse(
    array_1: NDArrayFloat,
    array_2: NDArrayFloat,
    axis: int | None = None,
    keepdims: bool = True,
) -> float | NDArrayFloat:
    """Compute the RMSE between `array_1` and `array_2`.

    Examples:
        >>> array_1 = [[1, 2], [4, 5]]
        >>> array_2 = [[3, 4], [5, 6]]
        >>> rmse(array_1, array_2, axis=None)
        1.5811388300841898
        >>> rmse(array_1, array_2, axis=0)
        array([[1.58113883, 1.58113883]])
        >>> rmse(array_1, array_2, axis=1)
        array([[2.],
               [1.]])
    """
    array_1 = np.asarray(array_1, dtype=float)
    array_2 = np.asarray(array_2, dtype=float)

    result: NDArrayFloat = np.sqrt(
        np.mean(np.square(array_1 - array_2), axis=axis, keepdims=keepdims),
    )

    if axis is None:
        return float(result)

    return result
