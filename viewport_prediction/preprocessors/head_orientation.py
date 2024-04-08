from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import json
from datetime import datetime

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path

    from viewport_prediction.types import NDArrayFloat


class HeadOrientationLog(NamedTuple):
    timestamps: NDArrayFloat

    list_deg_roll: NDArrayFloat
    list_deg_pitch: NDArrayFloat
    list_deg_yaw: NDArrayFloat

    list_rad_roll: NDArrayFloat
    list_rad_pitch: NDArrayFloat
    list_rad_yaw: NDArrayFloat

    list_quat_x: NDArrayFloat
    list_quat_y: NDArrayFloat
    list_quat_z: NDArrayFloat


def read_json(raw_file: Path, root_key: str) -> HeadOrientationLog:
    """Read raw head orientation log from the provided JSON file.

    Args:
        raw_file (Path): The JSON file to read data from.
        root_key (str): One of "1-SubheadPosition", "".

    Returns:
        Dict[str, NDArrayFloat]:
            keys = ["timestamps", "rolls", "pitches", "yaws", "quat_x", "quat_y",
                    "quat_z"]
    """
    timestamps = []
    list_deg_roll, list_deg_pitch, list_deg_yaw = [], [], []
    list_quat_x, list_quat_y, list_quat_z = [], [], []

    raw_data = json.loads(raw_file.read_text())[root_key]
    raw_timestamps = sorted(raw_data.keys())

    for raw_timestamp in raw_timestamps:
        timestamp = datetime.strptime(
            raw_timestamp.replace("_Z", "_0Z"),
            "%Y%m%dZ-%H%M%S_%fZ",
        ).timestamp()
        timestamps.append(timestamp)

        list_deg_roll.append(raw_data[raw_timestamp]["Ez"])
        list_deg_pitch.append(raw_data[raw_timestamp]["Ex"])
        list_deg_yaw.append(raw_data[raw_timestamp]["Ey"])

        # TODO: Remember to collect `quat_w` next time. # pylint: disable=fixme
        list_quat_x.append(raw_data[raw_timestamp]["Rz"])
        list_quat_y.append(raw_data[raw_timestamp]["Rx"])
        list_quat_z.append(raw_data[raw_timestamp]["Ry"])

    return HeadOrientationLog(
        timestamps=np.asarray(timestamps, dtype=float),
        #
        list_deg_roll=np.asarray(list_deg_roll, dtype=float),
        list_deg_pitch=np.asarray(list_deg_pitch, dtype=float),
        list_deg_yaw=np.asarray(list_deg_yaw, dtype=float),
        #
        list_rad_roll=np.radians(list_deg_roll, dtype=float),
        list_rad_pitch=np.radians(list_deg_pitch, dtype=float),
        list_rad_yaw=np.radians(list_deg_yaw, dtype=float),
        #
        list_quat_x=np.asarray(list_quat_x, dtype=float),
        list_quat_y=np.asarray(list_quat_y, dtype=float),
        list_quat_z=np.asarray(list_quat_z, dtype=float),
    )
