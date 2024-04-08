from __future__ import annotations

from typing import TYPE_CHECKING

import os
import functools
import subprocess
from multiprocessing import Pool

from rich.progress import Progress


if TYPE_CHECKING:
    from pathlib import Path

    from viewport_prediction.types import NDArrayFloat


def get_duration(video_file: Path) -> float:
    """Return video duration."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_file),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        check=False,
        encoding="utf8",
    )

    if process.returncode != 0:
        raise ValueError(f"failed to get video duration: {process.stderr}")

    duration = int(float(process.stdout) * 10) / 10
    return duration


def extract_frame(
    video_file: Path,
    playback_time: float,
    out_dir: Path,
    target_size: tuple[int, int],
    image_format: str = "jpg",
) -> None:
    """Save a frame from a video at provided playback time.

    Notes:
        No error will be returned when the provided playback time is negative or larger
        than video duration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target_height, target_width = target_size

    out_file = out_dir / f"frame_{playback_time:.2f}.{image_format}"
    if out_file.exists():
        return

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(playback_time),
        "-i",
        str(video_file),
        "-vframes",
        "1",
        "-f",
        "image2",
        "-vf",
        f"scale={target_width}:{target_height}",
        str(out_file),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        check=False,
        encoding="utf8",
    )

    if process.returncode != 0 or not out_file.exists():
        raise ValueError(
            f"failed to extract frame at {playback_time}s from {video_file}: "
            f"{process.stderr}",
        )


def _extract_frame_wrapper(
    video_file: Path,
    out_dir: Path,
    target_size: tuple[int, int],
    image_format: str,
    playback_time: float,
) -> None:
    extract_frame(video_file, playback_time, out_dir, target_size, image_format)


def extract_frames(
    video_file: Path,
    playback_time: NDArrayFloat,
    out_dir: Path,
    target_size: tuple[int, int],
    image_format: str = "jpg",
) -> None:
    n_processes = len(os.sched_getaffinity(0))

    with Pool(processes=n_processes) as pool, Progress() as progress:
        task = progress.add_task("Extracting frames...", total=playback_time.shape[0])

        for _ in enumerate(
            pool.imap_unordered(
                functools.partial(
                    _extract_frame_wrapper,
                    video_file,
                    out_dir,
                    target_size,
                    image_format,
                ),
                playback_time,
            ),
        ):
            progress.advance(task)
