from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat


def spherical_to_cartesian(
    inclination: NDArrayFloat,
    azimuth: NDArrayFloat,
    radius: NDArrayFloat | None = None,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Convert Spherical coordinates to Cartesian coordinates.

    Notes:
        If `radius` is not provided via the default `radius=None`, all points are assumed
        to be placed on the surface of a unit sphere.

    Examples:
        >>> inclination = [
        ...     [0.9553, 2.1862, 0.9553, 2.1862],
        ...     [0.9553, 2.1862, 0.9553, 2.1862],
        ... ]
        >>> azimuth = [
        ...     [0.7854, 0.7854, -0.7854, -0.7854],
        ...     [2.3562, 2.3562, -2.3562, -2.3562],
        ... ]
        >>> radius = r = np.sqrt(3) * np.ones((2, 4))
        >>> x, y, z = spherical_to_cartesian(inclination, azimuth, radius)
        >>> np.round(x)
        array([[ 1.,  1.,  1.,  1.],
               [-1., -1., -1., -1.]])
        >>> np.round(y)
        array([[ 1.,  1., -1., -1.],
               [ 1.,  1., -1., -1.]])
        >>> np.round(z)
        array([[ 1., -1.,  1., -1.],
               [ 1., -1.,  1., -1.]])
    """
    if radius is None:
        radius = np.ones_like(inclination)

    x = radius * np.cos(azimuth) * np.sin(inclination)
    y = radius * np.sin(azimuth) * np.sin(inclination)
    z = radius * np.cos(inclination)
    return (x, y, z)


def cartesian_to_spherical(
    x: NDArrayFloat,
    y: NDArrayFloat,
    z: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Convert Cartesian coordinates to Spherical coordinates.

    Examples:
        >>> x = [
        ...     [1, 1, 1, 1],
        ...     [-1, -1, -1, -1],
        ... ]
        >>> y = [
        ...     [1, 1, -1, -1],
        ...     [1, 1, -1, -1],
        ... ]
        >>> z = [
        ...     [1, -1, 1, -1],
        ...     [1, -1, 1, -1],
        ... ]
        >>> inclination, azimuth, radius = cartesian_to_spherical(x, y, z)
        >>> inclination
        array([[0.95531662, 2.18627604, 0.95531662, 2.18627604],
               [0.95531662, 2.18627604, 0.95531662, 2.18627604]])
        >>> azimuth
        array([[ 0.78539816,  0.78539816, -0.78539816, -0.78539816],
               [ 2.35619449,  2.35619449, -2.35619449, -2.35619449]])
        >>> radius
        array([[1.73205081, 1.73205081, 1.73205081, 1.73205081],
               [1.73205081, 1.73205081, 1.73205081, 1.73205081]])
    """
    radius = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    inclination = np.arccos(z / radius)
    azimuth = np.arctan2(y, x)
    return (inclination, azimuth, radius)


def euler_angles_to_spherical_coordinates(
    pitch: NDArrayFloat,
    yaw: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Convert Unity's Euler angles to Spherical coordinates.

    Notes:
        All points are assumed to be placed on the surface of a unit sphere.

    Examples
        >>> pitch = [0.0, np.pi/2, np.pi]
        >>> yaw = [0.0, np.pi/2, np.pi]
        >>> inclination, azimuth, radius = euler_angles_to_spherical_coordinates(
        ...     pitch, yaw
        ... )
        >>> inclination
        array([1.57079633, 3.14159265, 1.57079633])
        >>> azimuth
        array([6.28318531, 4.71238898, 3.14159265])
        >>> radius
        array([1., 1., 1.])
    """

    def _process_pitch(pitch: NDArrayFloat) -> NDArrayFloat:
        pitch = np.remainder(pitch + np.pi / 2, 2 * np.pi)
        pitch[pitch > np.pi] = 2 * np.pi - pitch[pitch > np.pi]
        return pitch

    def _process_yaw(yaw: NDArrayFloat) -> NDArrayFloat:
        yaw = 2 * np.pi - yaw
        return yaw

    inclination = _process_pitch(np.asarray(pitch, dtype=float))
    azimuth = _process_yaw(np.asarray(yaw, dtype=float))
    radius: NDArrayFloat = np.ones_like(azimuth, dtype=float)

    return (inclination, azimuth, radius)


def spherical_to_pixel(
    inclination: NDArrayFloat,
    azimuth: NDArrayFloat,
    frame_height: int,
    frame_width: int,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Convert Spherical coordinates to Pixel coordinates in a frame obtained by Equirectangular projection.

    Notes:
        The orgin of Pixel coordinates is in the top-left corner of the image.
        The first pixel coordinate `x` increases to the right, and `y` increases downwards.

    Examples:
        >>> x, y = spherical_to_pixel([0], [0], 2160, 3840)
        >>> x
        array([960.])
        >>> y
        array([0.])
    """  # pylint: disable=line-too-long # noqa: E501, D202, B950

    def process_azimuth(azimuth: NDArrayFloat) -> NDArrayFloat:
        temp: NDArrayFloat = np.copy(azimuth)

        indices = np.logical_and(azimuth >= 0, azimuth < np.pi / 2)
        temp[indices] = np.pi / 2 - azimuth[indices]

        indices = np.logical_and(azimuth >= np.pi / 2, azimuth <= 2 * np.pi)
        temp[indices] = -1 * azimuth[indices] + 5 * np.pi / 2

        temp_deg: NDArrayFloat = np.degrees(temp)
        return temp_deg

    vertical_angle: NDArrayFloat = np.pi / 2 - np.asarray(inclination, dtype=float)
    horizontal_angle = process_azimuth(np.asarray(azimuth, dtype=float))

    pos_y = frame_height / 2.0 - (frame_height / 2.0 * np.sin(vertical_angle))
    pos_x = horizontal_angle * 1.0 / 360.0 * frame_width

    return (pos_x, pos_y)
