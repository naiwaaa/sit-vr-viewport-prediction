from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from viewport_prediction.metrics.base_metric import BaseMetric


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


class RMSEMetric(BaseMetric):
    def __init__(self) -> None:
        self._sum_squared_error = 0.0
        self._count = 0

    def update(self, preds: NDArrayFloat, targets: NDArrayFloat) -> None:
        self._sum_squared_error += float(np.sum(np.square(preds - targets), axis=None))
        self._count += preds.flatten().shape[0]

    def compute(self) -> float:
        result = float(np.sqrt(self._sum_squared_error / self._count))
        return result

    def reset(self) -> None:
        self._sum_squared_error = 0.0
        self._count = 0
