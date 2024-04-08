from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from viewport_prediction.metrics.base_metric import BaseMetric
from viewport_prediction.helpers.orthodromic_distance import use_haversine


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


class OrthodromicDistanceMetric(BaseMetric):
    def __init__(self) -> None:
        self._all_distances: list[NDArrayFloat] = []

    def update(self, preds: NDArrayFloat, targets: NDArrayFloat) -> None:
        reshaped_preds = preds.reshape((-1, 2))
        reshaped_targets = targets.reshape((-1, 2))

        self._all_distances.append(
            use_haversine(
                inclination_1=reshaped_preds[:, 0],
                azimuth_1=reshaped_preds[:, 1],
                inclination_2=reshaped_targets[:, 0],
                azimuth_2=reshaped_targets[:, 1],
            ),
        )

    def compute(self) -> dict[str, float]:
        return {
            "mean": np.mean(self._all_distances),
            "std": np.std(self._all_distances),
        }

    def reset(self) -> None:
        self._all_distances = []
