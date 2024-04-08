from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import tensorflow as tf


if TYPE_CHECKING:
    from typing import Any

    from viewport_prediction.types import BatchData, DataLoader
    from viewport_prediction.entities.session import Session


class BatchIndex(NamedTuple):
    session_idx: int
    time_step: int


class BaseDataset:
    TfBatchDType: Any = (tf.float32, tf.float32)  # pylint: disable=invalid-name

    def __init__(
        self,
        sessions: list[Session],
        past_window_size: int,
        future_window_size: int,
    ) -> None:
        self.sessions = sessions
        self.past_window_size = past_window_size
        self.future_window_size = future_window_size

        self._batch_indices = self._build_indices()

        self.tf_dataset = tf.data.Dataset.range(  # pragma: no cover
            len(self._batch_indices),
        ).map(
            lambda x: tf_map(
                func=self.__getitem__,
                inp=x,
                tf_out_dtype=self.TfBatchDType,
            ),
        )

    def __len__(self) -> int:
        return len(self._batch_indices)

    def __getitem__(self, idx: int) -> BatchData:
        batch_idx = self._batch_indices[idx]
        return self.item(
            session=self.sessions[batch_idx.session_idx],
            time_step=batch_idx.time_step,
        )

    def item(self, session: Session, time_step: int) -> BatchData:
        raise NotImplementedError()

    def loader(self, batch_size: int) -> DataLoader:
        return (
            self.tf_dataset.shuffle(
                buffer_size=5000,
                reshuffle_each_iteration=True,
            )
            .batch(
                batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def _build_indices(self) -> list[BatchIndex]:
        batch_indices = []

        for session_idx, session in enumerate(self.sessions):
            # [0 1 2 3 4 5 6 7 8 9 10]
            # past_window_size = 4, future_window_size = 3
            # -> batch_indices = [4 5 6 7]
            batch_indices.extend(
                [
                    BatchIndex(session_idx, time_step)
                    for time_step in range(
                        self.past_window_size,
                        session.n_samples - self.future_window_size,
                    )
                ],
            )

        return batch_indices


def tf_map(func: Any, inp: Any, tf_out_dtype: Any) -> Any:  # pragma: no cover
    *x, y = tf.py_function(func=func, inp=[inp], Tout=tf_out_dtype)
    return tuple(x), y
