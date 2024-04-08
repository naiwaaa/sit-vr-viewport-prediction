#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from viewport_prediction import preprocessors
from viewport_prediction.utils import image, video, console, serializer, round_to_half
from viewport_prediction.helpers.coordinate_converter import (
    euler_angles_to_spherical_coordinates,
)


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat, NDArrayUInt8
    from viewport_prediction.preprocessors.subtitle import Subtitle


DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"
FRAME_SIZE = (224, 224)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.all:
        args.subtitle = True
        args.frames = True
        args.saliency = True

    videos_dir = DATA_DIR / "videos"
    video_indices = list(range(1, 6))

    console.print_divider("has_subtitle group")
    _run_group(
        args,
        group_dir=DATA_DIR / "has_subtitle",
        videos_dir=videos_dir,
        video_indices=video_indices,
    )

    console.print_divider("no_subtitle group")
    _run_group(
        args,
        group_dir=DATA_DIR / "no_subtitle",
        videos_dir=videos_dir,
        video_indices=video_indices,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess the collected data.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--subtitle", action="store_true")
    parser.add_argument(
        "--gpu_id",
        type=int,
        action="store",
        default=None,
        help="GPU used for PanoSalNet",
    )
    return parser


def _run_group(
    args: argparse.Namespace,
    group_dir: Path,
    videos_dir: Path,
    video_indices: list[int],
) -> None:
    n_sessions = len(list(group_dir.iterdir()))
    count = 0

    for video_id in video_indices:
        video_file = videos_dir / f"video_{video_id:02}.mp4"
        video_subtitles = (
            preprocessors.subtitle.read_srt(
                videos_dir / f"video_{video_id:02}_subtitle.srt",
            )
            if "has_subtitle" in str(group_dir)
            else None
        )

        for session_dir in group_dir.glob(f"video_{video_id:02}_user_*"):
            count += 1
            session_name = session_dir.name
            console.print(
                f"[{count:02}/{n_sessions:02}] "
                f"Preprocessing session {session_name}...",
            )

            _run_session(
                args,
                session_dir=session_dir,
                videos_dir=videos_dir,
                video_id=video_id,
                video_file=video_file,
                video_subtitles=video_subtitles,
                out_file=session_dir / "processed_data.csv",
            )


def _run_session(
    args: argparse.Namespace,
    session_dir: Path,
    videos_dir: Path,
    video_id: int,
    video_file: Path,
    video_subtitles: list[Subtitle] | None,
    out_file: Path,
) -> None:
    """Preprocess the collected data for each session."""
    video_duration = video.get_duration(video_file)
    has_subtitle = video_subtitles is not None

    # ====================
    # head orientation log
    # ====================
    processed_data = pd.DataFrame.from_dict(
        _preprocess_head_orientation_log(
            session_dir / "raw_head_orientation_log.json",
            f"{video_id}-SubheadPosition"
            if has_subtitle
            else f"{video_id}-NonheadPosition",
        ),
    )

    processed_data.sort_values("timestamps", inplace=True, ignore_index=True)

    processed_data.drop(
        processed_data[processed_data.playback_time >= video_duration].index,
        inplace=True,
    )

    assert np.all(
        processed_data.rounded_playback_time[:-1].to_numpy()
        < processed_data.rounded_playback_time[1:].to_numpy(),
    )
    assert np.allclose(
        a=processed_data.rounded_playback_time[1:].to_numpy()
        - processed_data.rounded_playback_time[:-1].to_numpy(),
        b=0.5,
    )

    # ============
    # subtitles
    # ============
    if args.subtitle and video_subtitles is not None:
        processed_data["subtitle"] = _preprocess_subtitle(
            video_subtitles,
            processed_data.playback_time,
        )

    # ============
    # video frames
    # ============
    if args.frames:
        _preprocess_video_frames(
            playback_time=processed_data.rounded_playback_time,
            video_id=video_id,
            videos_dir=videos_dir,
            video_file=video_file,
        )

    # ============
    # video frames
    # ============
    processed_data.to_csv(out_file, index=False)


def _preprocess_head_orientation_log(
    raw_file: Path,
    root_key: str,
) -> dict[str, NDArrayFloat]:
    log = preprocessors.head_orientation.read_json(raw_file, root_key)

    playback_time = log.timestamps - log.timestamps[1]
    rounded_playback_time = round_to_half(playback_time)

    inclination, azimuth, _ = euler_angles_to_spherical_coordinates(
        log.list_rad_pitch,
        log.list_rad_yaw,
    )

    data = {
        "timestamps": log.timestamps[1:],
        "playback_time": playback_time[1:],
        "rounded_playback_time": rounded_playback_time[1:],
        #
        "quat_x": log.list_quat_x[1:],
        "quat_y": log.list_quat_y[1:],
        "quat_z": log.list_quat_z[1:],
        #
        "deg_roll": log.list_deg_roll[1:],
        "deg_pitch": log.list_deg_pitch[1:],
        "deg_yaw": log.list_deg_yaw[1:],
        #
        "rad_roll": log.list_rad_roll[1:],
        "rad_pitch": log.list_rad_pitch[1:],
        "rad_yaw": log.list_rad_yaw[1:],
        #
        "deg_inclination": np.degrees(inclination[1:]),
        "deg_azimuth": np.degrees(azimuth[1:]),
        #
        "rad_inclination": inclination[1:],
        "rad_azimuth": azimuth[1:],
    }

    return data


def _preprocess_subtitle(
    subtitles: list[Subtitle],
    playback_time: NDArrayFloat,
) -> list[str]:
    length = playback_time.shape[0]
    processed_subtitle = [""] * length

    last_idx = 0
    for subtitle in subtitles:
        for idx in range(last_idx, length):
            if subtitle.start_time <= playback_time[idx] <= subtitle.end_time:
                processed_subtitle[idx] = subtitle.content
            elif playback_time[idx] > subtitle.end_time:
                last_idx = idx
                break

    return processed_subtitle


def _preprocess_video_frames(
    playback_time: NDArrayFloat,
    video_id: int,
    videos_dir: Path,
    video_file: Path,
) -> None:
    frames_dir = videos_dir / f"video_{video_id:02}_frames"
    frames_npy_file = videos_dir / f"video_{video_id:02}_frames.npy"

    video.extract_frames(
        video_file,
        playback_time,
        out_dir=frames_dir,
        target_size=FRAME_SIZE,
        image_format="jpg",
    )

    if not frames_npy_file.exists():
        frames = []
        for time_step in playback_time:
            frame_file = frames_dir / f"frame_{time_step:.2f}.jpg"
            frames.append(image.imread(frame_file, "rgb"))

        serializer.serialize(frames_npy_file, np.asarray(frames, dtype=np.uint8))

    npy_frames: NDArrayUInt8 = serializer.deserialize(frames_npy_file)
    assert npy_frames.shape[0] == playback_time.shape[0]
    assert npy_frames.shape[1] == FRAME_SIZE[0]
    assert npy_frames.shape[2] == FRAME_SIZE[1]
    assert npy_frames.shape[3] == 3


if __name__ == "__main__":
    main()
