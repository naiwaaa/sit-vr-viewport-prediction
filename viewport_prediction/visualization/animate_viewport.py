from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


if TYPE_CHECKING:
    from typing import Any

    from pathlib import Path

    from matplotlib.axes import Axes

    from viewport_prediction.types import NDArrayInt, NDArrayFloat, NDArrayUInt8


class ViewportAnimationArtists(NamedTuple):
    actual_viewport: Any
    actual_horizontal_movement: Any
    predicted_viewport: Any | None
    predicted_horizontal_movement: Any | None


def animate_viewport(  # noqa: C901
    frames: NDArrayUInt8,
    playback_time: NDArrayFloat,
    actual_spherical_coords: NDArrayFloat,
    predicted_spherical_coords: NDArrayFloat | None = None,
    n_history_steps: int = 20,
    subplot_width: int = 5,
    out_mp4_file: Path | None = None,
) -> animation.FuncAnimation:
    n_cols = 2
    n_rows = 2 if predicted_spherical_coords is not None else 1

    fig, list_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * subplot_width, n_rows * subplot_width),
        constrained_layout=True,
    )
    list_axes = np.asarray(list_axes).flatten()

    # ===========
    # init figure
    # ===========
    artists = _init_all_axes(
        actual_viewport_axes=list_axes[0],
        actual_horizontal_movement_axes=list_axes[1],
        predicted_viewport_axes=None if n_rows == 1 else list_axes[2],
        predicted_horizontal_movement_axes=None if n_rows == 1 else list_axes[3],
        n_history_steps=n_history_steps,
    )

    def init_func() -> list[Any]:
        return [artist for artist in artists if artist is not None]

    # ===============
    # update function
    # ===============
    update_func = partial(
        _update_all_axes,
        frames=frames,
        actual_spherical_coords=actual_spherical_coords,
        predicted_spherical_coords=predicted_spherical_coords,
        n_history_steps=n_history_steps,
        artists=artists,
    )

    ani = animation.FuncAnimation(
        fig,
        update_func,
        init_func=init_func,
        interval=1000,
        frames=range(playback_time.shape[0]),
        blit=True,
    )

    if out_mp4_file is not None:
        ani.save(out_mp4_file, writer="ffmpeg")

    return ani


def _init_all_axes(
    actual_viewport_axes: Axes,
    actual_horizontal_movement_axes: Axes | None = None,
    predicted_viewport_axes: Axes | None = None,
    predicted_horizontal_movement_axes: Axes | None = None,
    n_history_steps: int = 20,
) -> ViewportAnimationArtists:
    actual_viewport_artist = _init_viewport_axes(
        actual_viewport_axes,
        title="Actual user viewport",
    )

    actual_horizontal_movement_artist = _init_horizontal_movement_axes(
        actual_horizontal_movement_axes,
        title="Actual user horizontal movement",
        n_history_steps=n_history_steps,
    )

    predicted_viewport_artist = (
        None
        if predicted_viewport_axes is None
        else _init_viewport_axes(predicted_viewport_axes, title="Predicted user viewport")
    )

    predicted_horizontal_movement_artist = (
        None
        if predicted_horizontal_movement_axes is None
        else _init_horizontal_movement_axes(
            predicted_horizontal_movement_axes,
            title="Predicted user horizontal movement",
            n_history_steps=n_history_steps,
        )
    )

    return ViewportAnimationArtists(
        actual_viewport_artist,
        actual_horizontal_movement_artist,
        predicted_viewport_artist,
        predicted_horizontal_movement_artist,
    )


def _init_viewport_axes(axes: Axes, title: str) -> plt.Line2D:
    axes.axis("off")
    axes.title.set_text(title)

    return axes.imshow(np.zeros((1, 1, 3)))


def _init_horizontal_movement_axes(
    axes: Axes,
    title: str,
    n_history_steps: int,
) -> plt.Line2D:
    axes.set_title(title)
    axes.set_xlim((-360, 360))
    axes.set_ylim((0, n_history_steps + 1))
    axes.invert_yaxis()
    axes.xaxis.tick_top()

    (axes_data,) = axes.plot([], [], marker="o")

    return axes_data


def _update_all_axes(
    step: int,
    frames: NDArrayUInt8,
    actual_spherical_coords: NDArrayFloat,
    predicted_spherical_coords: NDArrayFloat | None,
    n_history_steps: int,
    artists: ViewportAnimationArtists,
) -> list[Any]:
    frame: NDArrayUInt8 = frames[step]

    history_actual_coords = actual_spherical_coords[
        max(0, step - n_history_steps) : (step + 1),  # noqa: E203
    ]

    _update_viewport_artist(
        artists.actual_viewport,
        frame,
        history_actual_coords[-1, 0],
        history_actual_coords[-1, 1],
    )
    _update_horizontal_movement_artist(
        artists.actual_horizontal_movement,
        history_actual_coords[:, 1],
    )

    if predicted_spherical_coords is not None:
        history_predicted_coords = predicted_spherical_coords[
            max(0, step - n_history_steps) : (step + 1),  # noqa: E203
        ]

        _update_viewport_artist(
            artists.actual_viewport,
            frame,
            history_predicted_coords[-1, 0],
            history_predicted_coords[-1, 1],
        )
        _update_horizontal_movement_artist(
            artists.predicted_horizontal_movement,
            history_predicted_coords[:, 1],
        )

    return [artist for artist in artists if artist is not None]


def _update_viewport_artist(
    artist: Any,
    frame: NDArrayUInt8,
    inclination: float,
    azimuth: float,
) -> plt.Line2D:
    artist.set_data(
        _crop_viewport_from_frame(
            frame,
            inclination,
            azimuth,
            viewport_width=360,
        ),
    )


def _update_horizontal_movement_artist(
    artist: Any,
    history_azimuths: NDArrayFloat,
) -> plt.Line2D:
    artist.set_data(
        np.degrees(3 * np.pi / 2 - history_azimuths),
        range(history_azimuths.shape[0]),
    )


def _crop_viewport_from_frame(
    frame: NDArrayUInt8,
    inclination: float,
    azimuth: float,
    viewport_width: float,
    fov_v: float = np.pi / 2,
    fov_h: float = np.pi / 2,
) -> NDArrayUInt8:
    # pylint: disable=invalid-name, too-many-locals
    def get_rotation_x(rad: float) -> NDArrayFloat:
        cos_value = np.cos(rad)
        sin_value = np.sin(rad)
        return np.array(
            [
                [1, 0, 0],
                [0, cos_value, sin_value],
                [0, -sin_value, cos_value],
            ],
        )

    def get_rotation_y(rad: float) -> NDArrayFloat:
        cos_value = np.cos(rad)
        sin_value = np.sin(rad)
        return np.array(
            [
                [cos_value, 0, sin_value],
                [0, 1, 0],
                [-sin_value, 0, cos_value],
            ],
        )

    (frame_height, frame_width, _) = frame.shape
    viewport_height = int(viewport_width * np.tan(fov_v / 2) / np.tan(fov_h / 2))
    viewport_image: NDArrayUInt8 = np.zeros(
        (viewport_height, viewport_width, 3),
        dtype=np.uint8,
    )

    inclination = np.pi / 2 - inclination
    azimuth = 3 * np.pi / 2 - azimuth

    matrix = np.dot(
        get_rotation_y(azimuth),
        get_rotation_x(inclination),
    )

    DI: NDArrayInt = np.ones(  # noqa: N806
        (viewport_width * viewport_height, 3),
        dtype=int,
    )
    trans: NDArrayFloat = np.array(
        [
            [
                2.0 * np.tan(fov_h / 2) / float(viewport_width),
                0.0,
                -np.tan(fov_h / 2),
            ],
            [
                0.0,
                -2.0 * np.tan(fov_v / 2) / float(viewport_height),
                np.tan(fov_v / 2),
            ],
        ],
        dtype=float,
    )

    xx, yy = np.meshgrid(np.arange(viewport_width), np.arange(viewport_height))
    DI[:, 0] = xx.reshape(viewport_width * viewport_height)
    DI[:, 1] = yy.reshape(viewport_width * viewport_height)

    v: NDArrayFloat = np.ones((viewport_width * viewport_height, 3), dtype=float)

    v[:, :2] = np.dot(DI, trans.T)
    v = np.dot(v, matrix.T)

    diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
    theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
    phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

    ey = np.rint(theta * frame_height / np.pi).astype(int)
    ex = np.rint(phi * frame_width / (2 * np.pi)).astype(int)

    ex[ex >= frame_width] = frame_width - 1
    ey[ey >= frame_height] = frame_height - 1

    viewport_image[DI[:, 1], DI[:, 0]] = frame[ey, ex]

    return viewport_image
