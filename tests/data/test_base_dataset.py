from __future__ import annotations

from typing import TYPE_CHECKING

from viewport_prediction.data import BaseDataset
from viewport_prediction.entities import Session


if TYPE_CHECKING:
    from pathlib import Path


def test_create_base_dataset(has_subtitle_session_dir: Path) -> None:
    sessions = [Session(has_subtitle_session_dir)]

    BaseDataset(sessions, past_window_size=5, future_window_size=5)  # act
