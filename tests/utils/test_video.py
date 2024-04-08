from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import numpy as np

from viewport_prediction.utils import video


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow()
def test_get_duration(realshort_video_file: Path) -> None:
    result = video.get_duration(realshort_video_file)

    assert result == 1.1


@pytest.mark.slow()
def test_get_duration_should_fail_when_file_not_exists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed to get video duration"):
        video.get_duration(tmp_path)


@pytest.mark.slow()
def test_extract_frame(tmp_path: Path, realshort_video_file: Path) -> None:
    video.extract_frame(
        realshort_video_file,
        playback_time=0.5,
        out_dir=tmp_path,
        target_size=(300, 400),
    )

    result = tmp_path / "frame_0.50.jpg"

    assert result.exists()
    assert result.is_file()


@pytest.mark.slow()
def test_extract_frame_should_skip_when_file_exists(
    tmp_path: Path,
    realshort_video_file: Path,
) -> None:
    file = tmp_path / "frame_0.50.jpg"
    video.extract_frame(
        realshort_video_file,
        playback_time=0.5,
        out_dir=tmp_path,
        target_size=(300, 400),
    )
    mtime = file.stat().st_mtime
    video.extract_frame(
        realshort_video_file,
        playback_time=0.5,
        out_dir=tmp_path,
        target_size=(300, 400),
    )

    result = file.stat().st_mtime

    assert mtime == result


@pytest.mark.slow()
def test_extract_frame_should_fail_when_file_not_exists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed to extract frame"):
        video.extract_frame(
            tmp_path,
            playback_time=0.5,
            out_dir=tmp_path,
            target_size=(300, 400),
        )


@pytest.mark.slow()
def test_extract_frames(tmp_path: Path, realshort_video_file: Path) -> None:
    video.extract_frames(
        realshort_video_file,
        playback_time=np.asarray([0.1, 0.2]),
        out_dir=tmp_path,
        target_size=(300, 400),
    )

    result = [
        tmp_path / "frame_0.10.jpg",
        tmp_path / "frame_0.20.jpg",
    ]

    assert all(file.exists() for file in result)
