from __future__ import annotations

from typing import TYPE_CHECKING

from viewport_prediction.data import HFDataset
from viewport_prediction.entities import Session


if TYPE_CHECKING:
    from pathlib import Path


def test_create_hf_dataset(has_subtitle_session_dir: Path) -> None:
    sessions = [Session(has_subtitle_session_dir)]

    dataset = HFDataset(sessions, past_window_size=5, future_window_size=3)  # act

    assert len(dataset) == 150
    assert dataset[0][0].shape == (5, 2)  # head_orientation_input
    assert dataset[0][1].shape == (5, 224, 224, 3)  # frame_input
    assert dataset[0][2].shape == (3, 2)  # head_orientation_output


def test_hf_dataset_load_batches(has_subtitle_session_dir: Path) -> None:
    sessions = [Session(has_subtitle_session_dir)]
    dataset = HFDataset(sessions, past_window_size=5, future_window_size=3)

    all_batches = list(dataset.loader(batch_size=32).as_numpy_iterator())  # act

    assert len(all_batches) == 4
    for batch in all_batches:
        (head_orientation_input, frame_input), head_orientation_output = batch
        #
        assert head_orientation_input.shape == (32, 5, 2)
        assert frame_input.shape == (32, 5, 224, 224, 3)
        assert head_orientation_output.shape == (32, 3, 2)
