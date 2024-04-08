from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import numpy as np

from viewport_prediction.utils import serializer


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("file_name", ["data.npy", "data.pkl", "data.json"])
def test_deserialize_inverts_serialize(tmp_path: Path, file_name: str) -> None:
    data = [0, 1, 2]
    file = tmp_path / file_name
    serializer.serialize(file, data)

    result = serializer.deserialize(file)

    assert np.array_equal(data, result)


def test_serialize_should_raise_value_error_with_unsupported_file_format(
    tmp_path: Path,
) -> None:
    data = [0, 1, 2]
    file = tmp_path / "data.random"

    with pytest.raises(ValueError, match="`.random` is not supported"):
        serializer.serialize(file, data)


def test_deserialize_should_raise_value_error_with_unsupported_file_format(
    tmp_path: Path,
) -> None:
    file = tmp_path / "data.random"

    with pytest.raises(ValueError, match="`.random` is not supported"):
        serializer.deserialize(file)
