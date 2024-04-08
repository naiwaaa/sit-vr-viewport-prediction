from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import tensorflow as tf

from viewport_prediction.data.base_dataset import BaseDataset


if TYPE_CHECKING:
    from viewport_prediction.types import BatchData
    from viewport_prediction.entities.session import Session


class HFDataset(BaseDataset):
    """Head orientation + Frame."""

    TfBatchDType = (tf.float32, tf.uint8, tf.float32)

    def item(self, session: Session, time_step: int) -> BatchData:
        start = time_step - self.past_window_size
        end = time_step + self.future_window_size

        x = (
            session.spherical_coords[start:time_step] / [np.pi, 2 * np.pi],
            session.frames[start:time_step] / 255.0,
        )
        y = session.spherical_coords[time_step:end]

        return [*x, y]
