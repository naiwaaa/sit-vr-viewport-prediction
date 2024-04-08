from __future__ import annotations

from typing import TYPE_CHECKING

from functools import cached_property

import pandas as pd

from viewport_prediction.utils import serializer


if TYPE_CHECKING:
    from pathlib import Path

    from viewport_prediction.types import NDArrayStr, NDArrayFloat, NDArrayUInt8


class Session:
    """A class to represent the collected data in one session (1 user - 1 video)."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir.resolve(strict=True)
        self.videos_dir = self.session_dir.parent.parent / "videos"
        self.session_name = self.session_dir.name

        self.processed_data = pd.read_csv(
            self.session_dir / "processed_data.csv",
            na_filter=False,
        )

        self.n_samples = self.playback_time.shape[0]
        self.has_subtitle = "has_subtitle" in str(self.session_dir)

    @cached_property
    def playback_time(self) -> NDArrayFloat:
        """Return the array of playback time."""
        result: NDArrayFloat = self.processed_data["rounded_playback_time"].to_numpy()
        return result

    @cached_property
    def spherical_coords(self) -> NDArrayFloat:
        """Return the array of head orientation data.

        Returns:
            NDArrayFloat: shape = `(n_samples, 2)`.
                Each sample is represented as a tuple of `(inclination, azimuth)`
        """
        result: NDArrayFloat = self.processed_data[
            ["rad_inclination", "rad_azimuth"]
        ].to_numpy()
        return result

    @cached_property
    def inclination(self) -> NDArrayFloat:
        result: NDArrayFloat = self.processed_data["rad_inclination"].to_numpy()
        return result

    @cached_property
    def azimuth(self) -> NDArrayFloat:
        result: NDArrayFloat = self.processed_data["rad_azimuth"].to_numpy()
        return result

    @cached_property
    def subtitle(self) -> NDArrayStr | None:
        if not self.has_subtitle:
            return None  # pragma: no cover

        result: NDArrayStr = self.processed_data.subtitle.to_numpy()

        return result

    @cached_property
    def frames(self) -> NDArrayUInt8:
        """Return the array of video frames.

        Returns:
            NDArrayUInt8: shape = `(n_samples, frame_height, frame_width, 3)`.
        """
        video_id, _ = _parse_session_name(self.session_name)

        frames: NDArrayUInt8 = serializer.deserialize(
            self.videos_dir / f"video_{video_id:02}_frames.npy",
        )

        return frames

    @cached_property
    def gts(self) -> NDArrayFloat:
        """Return the array of ground-truth saliency maps.

        Returns:
            NDArrayFloat: shape = `(n_samples, maps_height, map_width)`.
        """
        video_id, _ = _parse_session_name(self.session_name)

        gts: NDArrayFloat = serializer.deserialize(
            self.videos_dir / f"video_{video_id:02}_gts.npy",
        )

        return gts


def _parse_session_name(session_name: str) -> tuple[int, int]:
    """Return the video id and user id from a session name.

    Args:
        session_name (str): The name of the session.

    Returns:
        Tuple[int, int]: A tuple of video id and user id.

    Examples:
        >>> _parse_session_name("video_01_user_01")
        (1, 1)
        >>> _parse_session_name("video_03_user_15")
        (3, 15)
    """
    parts = session_name.split("_")
    video_id = int(parts[-3])
    user_id = int(parts[-1])
    return video_id, user_id
