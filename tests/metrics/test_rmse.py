from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from viewport_prediction.utils import rmse
from viewport_prediction.metrics import RMSEMetric


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


def test_rmse_metric() -> None:
    array_1: NDArrayFloat = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float)
    array_2: NDArrayFloat = np.asarray([[7, 8], [8, 9], [9, 10]], dtype=float)
    rmse_metric = RMSEMetric()
    for pred, target in zip(array_1, array_2):
        rmse_metric.update(pred, target)

    result = rmse_metric.compute()

    assert result == rmse(array_1, array_2)


def test_reset_rmse_metric() -> None:
    rmse_metric = RMSEMetric()
    rmse_metric.update(np.asarray([1]), np.asarray([2]))

    rmse_metric.reset()  # act

    # pylint: disable=protected-access
    assert rmse_metric._sum_squared_error == 0.0
    assert rmse_metric._count == 0
