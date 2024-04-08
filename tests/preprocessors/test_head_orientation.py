from __future__ import annotations

from typing import TYPE_CHECKING

from viewport_prediction.preprocessors import head_orientation


if TYPE_CHECKING:
    from pathlib import Path


def test_read_head_orientation_json(raw_head_orientation_log_file: Path) -> None:
    result = head_orientation.read_json(
        raw_file=raw_head_orientation_log_file,
        root_key="1-SubheadPosition",
    )

    assert result.timestamps.shape == result.list_deg_roll.shape
    assert result.timestamps.shape == result.list_rad_roll.shape
    assert result.timestamps.shape == result.list_quat_x.shape
    assert (
        result.list_deg_roll.shape
        == result.list_deg_pitch.shape
        == result.list_deg_yaw.shape
    )
    assert (
        result.list_rad_roll.shape
        == result.list_rad_pitch.shape
        == result.list_rad_yaw.shape
    )
    assert (
        result.list_quat_x.shape == result.list_quat_y.shape == result.list_quat_z.shape
    )
