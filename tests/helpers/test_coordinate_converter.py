from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from hypothesis.extra.numpy import arrays

import numpy as np

from viewport_prediction.helpers import coordinate_converter


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


@given(
    arrays(dtype=float, shape=(1, 2), elements=st.floats(-4 * np.pi, 4 * np.pi)),
    arrays(dtype=float, shape=(1, 2), elements=st.floats(-4 * np.pi, 4 * np.pi)),
)
@pytest.mark.slow()
def test_spherical_cartesian(inclination: NDArrayFloat, azimuth: NDArrayFloat) -> None:
    x, y, z = coordinate_converter.spherical_to_cartesian(inclination, azimuth)
    spherical_coords = coordinate_converter.cartesian_to_spherical(x, y, z)

    result = coordinate_converter.spherical_to_cartesian(
        spherical_coords[0],
        spherical_coords[1],
        spherical_coords[2],
    )

    assert_allclose(x, result[0], rtol=1e-5, atol=1e-5)
    assert_allclose(y, result[1], rtol=1e-5, atol=1e-5)
    assert_allclose(z, result[2], rtol=1e-5, atol=1e-5)
