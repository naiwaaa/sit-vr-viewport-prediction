from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import srt


if TYPE_CHECKING:
    from pathlib import Path


class Subtitle(NamedTuple):
    start_time: float
    end_time: float
    content: str


def read_srt(srt_file: Path) -> list[Subtitle]:
    """Read subtitles from the specified srt file."""
    subtitles = list(
        srt.sort_and_reindex(srt.parse(srt_file.read_text(encoding="utf-8"))),
    )

    return [
        Subtitle(
            start_time=sub.start.total_seconds(),
            end_time=sub.end.total_seconds(),
            content=" ".join(sub.content.splitlines()),
        )
        for sub in subtitles
    ]
