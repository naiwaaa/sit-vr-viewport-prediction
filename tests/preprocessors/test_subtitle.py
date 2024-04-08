from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from viewport_prediction.preprocessors import subtitle


if TYPE_CHECKING:
    from pathlib import Path


def test_read_srt(subtitle_file: Path) -> None:
    result = subtitle.read_srt(subtitle_file)

    assert len(result) == 2
    assert result[0].start_time == pytest.approx(9.64)
    assert result[0].end_time == pytest.approx(11.020)
    assert result[0].content == "Welcome to London!"
