from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


class BaseMetric:
    def update(self, preds: NDArrayFloat, targets: NDArrayFloat) -> None:
        raise NotImplementedError()

    def compute(self) -> float | dict[str, float]:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()
