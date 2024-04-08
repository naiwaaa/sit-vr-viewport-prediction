from __future__ import annotations

from typing import TYPE_CHECKING

from viewport_prediction.preprocessors import saliency
from viewport_prediction.entities.session import Session


if TYPE_CHECKING:
    from pathlib import Path


def test_extract_one_session(has_subtitle_session_dir: Path) -> None:
    session = Session(has_subtitle_session_dir)
    saliency_map_size = (125, 250)

    result = saliency.extract_one_session(session, saliency_map_size, save_to_file=False)

    assert result.shape[0] == session.playback_time.shape[0]
    assert result.shape[1] == saliency_map_size[0]
    assert result.shape[2] == saliency_map_size[1]


def test_extract_sessions(has_subtitle_session_dir: Path) -> None:
    sessions = [Session(has_subtitle_session_dir), Session(has_subtitle_session_dir)]
    saliency_map_size = (125, 250)

    result = saliency.extract_sessions(sessions, saliency_map_size, save_to_file=False)

    assert result.shape[0] == sessions[0].playback_time.shape[0]
    assert result.shape[1] == saliency_map_size[0]
    assert result.shape[2] == saliency_map_size[1]
