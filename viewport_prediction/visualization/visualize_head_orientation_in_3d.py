from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from viewport_prediction.helpers import coordinate_converter


if TYPE_CHECKING:
    from typing import Any, Literal

    from viewport_prediction.types import NDArrayFloat


def visualize_head_orientation_in_3d(
    inclination: NDArrayFloat,
    azimuth: NDArrayFloat,
    playback_time: NDArrayFloat | None = None,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
) -> go.Figure:
    x, y, z = coordinate_converter.spherical_to_cartesian(inclination, azimuth)

    fig = go.Figure()
    fig.add_traces(_sphere_grid_traces())
    fig.add_traces(_head_orientation_traces(x, y, z, playback_time, mode))
    fig.update_layout(
        margin={"l": 5, "r": 5, "t": 20, "b": 20},
    )

    return fig


def _sphere_grid_traces() -> list[Any]:
    # pylint: disable=invalid-name
    u, v = np.mgrid[0 : 2 * np.pi : 19j, 0 : np.pi : 9j]  # type: ignore # noqa
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    sphere = [
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.5,
            colorscale=[[0, "gray"], [0.5, "white"], [1, "gray"]],
            showscale=False,
            hoverinfo="none",
            showlegend=False,
        ),
    ]

    line_style = {
        "color": "black",
        "width": 1,
    }

    # vertical lines
    for i, j, k in zip(x, y, z):
        sphere.append(
            go.Scatter3d(
                x=i,
                y=j,
                z=k,
                mode="lines",
                line=line_style,
                hoverinfo="none",
                showlegend=False,
            ),
        )

    # horizontal lines
    for i, j, k in zip(x.T, y.T, z.T):
        sphere.append(
            go.Scatter3d(
                x=i,
                y=j,
                z=k,
                mode="lines",
                line=line_style,
                hoverinfo="none",
                showlegend=False,
            ),
        )

    return sphere


def _head_orientation_traces(
    x: NDArrayFloat,
    y: NDArrayFloat,
    z: NDArrayFloat,
    playback_time: NDArrayFloat | None = None,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
) -> Any:
    color_scale = (
        {}
        if playback_time is None
        else {
            "colorscale": "aggrnyl",
            "color": playback_time,
            "colorbar": {
                "title": "Playback time",
                "tickvals": [playback_time[0], playback_time[-1]],
            },
        }
    )
    line_style: dict[str, Any] = {**color_scale, "width": 5}
    marker_style: dict[str, Any] = {**color_scale, "size": 5}

    head_orientation_trace = (
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode=mode,
            name="Head orientation",
            showlegend=False,
            marker=marker_style,
            line=line_style,
        ),
    )

    return head_orientation_trace
