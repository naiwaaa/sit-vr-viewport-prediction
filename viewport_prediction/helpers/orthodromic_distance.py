from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


def use_haversine(
    inclination_1: NDArrayFloat,
    azimuth_1: NDArrayFloat,
    inclination_2: NDArrayFloat,
    azimuth_2: NDArrayFloat,
) -> NDArrayFloat:
    """Compute the Orhodromic distance.

    Orthodromic distance is the shortest angular distance between two points on the
    surface of a unit sphere. Each point is represented in Spherical coordinates
    `(inclination, azimuth)`.

    The distance is computed using the Haversine formula.

    Examples:
        >>> array_1 = np.radians([[60, 0], [39, 0]])
        >>> array_2 = np.radians([[90, 0], [124, -60]])
        >>> use_haversine(array_1[:, 0], array_1[:, 1], array_2[:, 0], array_2[:, 1])
        array([0.52359878, 1.7453914 ])
    """
    lat_1 = np.pi / 2 - inclination_1
    long_1 = azimuth_1
    lat_2 = np.pi / 2 - inclination_2
    long_2 = azimuth_2

    sin_diff_lat = np.sin((lat_1 - lat_2) / 2.0)
    sin_diff_long = np.sin((long_1 - long_2) / 2.0)

    distance: NDArrayFloat = 2 * np.arcsin(
        np.sqrt(
            sin_diff_lat * sin_diff_lat
            + np.cos(lat_1) * np.cos(lat_2) * sin_diff_long * sin_diff_long,
        ),
    )

    return distance
