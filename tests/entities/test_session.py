from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.testing import assert_allclose

from viewport_prediction.entities import Session


if TYPE_CHECKING:
    from pathlib import Path


def test_session(has_subtitle_session_dir: Path) -> None:
    result = Session(has_subtitle_session_dir)

    assert result.playback_time.shape == (158,)
    assert result.subtitle is not None
    assert result.subtitle.shape == (158,)


def test_session_head_orientation(has_subtitle_session_dir: Path) -> None:
    result = Session(has_subtitle_session_dir)

    assert result.playback_time.shape == (158,)
    assert result.spherical_coords.shape == (158, 2)
    assert_allclose(result.spherical_coords[:, 0], result.inclination)
    assert_allclose(result.spherical_coords[:, 1], result.azimuth)


def test_session_video_frames(has_subtitle_session_dir: Path) -> None:
    result = Session(has_subtitle_session_dir)

    assert result.frames.shape == (317, 224, 224, 3)


def test_session_gts(has_subtitle_session_dir: Path) -> None:
    result = Session(has_subtitle_session_dir)

    assert result.gts.shape == (317, 224, 224)
