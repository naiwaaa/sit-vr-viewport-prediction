import pytest

import numpy as np

from viewport_prediction.helpers import orthodromic_distance


@pytest.mark.parametrize(
    ("array_1", "array_2", "expected"),
    [
        (
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [0.0],
        ),
        (
            [[0.0, 0.0], [0.0, 0.0]],
            [[90.0, 0.0], [180.0, 0.0]],
            [90.0, 180.0],
        ),
    ],
)
def test_orthodormic_distance_use_haversine(
    array_1: list[float],
    array_2: list[float],
    expected: list[float],
) -> None:
    rad_array_1 = np.radians(array_1)
    rad_array_2 = np.radians(array_2)
    rad_expected = np.radians(expected)

    result = orthodromic_distance.use_haversine(
        inclination_1=rad_array_1[:, 0],
        azimuth_1=rad_array_1[:, 1],
        inclination_2=rad_array_2[:, 0],
        azimuth_2=rad_array_2[:, 1],
    )

    assert np.allclose(result, rad_expected)
