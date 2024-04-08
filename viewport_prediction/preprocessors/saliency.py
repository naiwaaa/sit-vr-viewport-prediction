from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from viewport_prediction.utils import image, console, serializer
from viewport_prediction.helpers import orthodromic_distance


if TYPE_CHECKING:
    from viewport_prediction.types import NDArrayFloat
    from viewport_prediction.entities.session import Session


def extract_one_session(
    session: Session,
    saliency_map_size: tuple[int, int],
    save_to_file: bool,
) -> NDArrayFloat:
    r"""Extract the ground-truth saliency maps on one session.

    $$\text{GT_Saliency}^t_{v,u} = $$

    Args:
        session (Session): A session.
        saliency_map_size (Tuple[int, int]): A tuple of `(height, width)`.
        save_to_file (bool): Allow saving the created saliency maps to `saliency_maps`
            directory in the `session.session_dir` directory.

    Returns:
        NDArrayFloat: shape = `(n_samples, saliency_map_height, saliency_map_width)`.
    """
    maps = []
    map_height, map_width = saliency_map_size

    saliency_maps_dir = session.session_dir / "saliency_maps"
    saliency_maps_dir.mkdir(parents=True, exist_ok=True)

    inclination_grid, azimuth_grid = np.meshgrid(
        np.linspace(0, np.pi, map_height, endpoint=False),
        np.linspace(0, 2 * np.pi, map_width, endpoint=False),
        indexing="ij",
    )
    inclination_grid, azimuth_grid = inclination_grid.flatten(), azimuth_grid.flatten()

    for timestep, viewport_spherical_coords in zip(
        session.playback_time,
        session.spherical_coords,
    ):
        # compute orthodromic distance
        tiled_viewport_spherical_coords: NDArrayFloat = np.tile(
            viewport_spherical_coords,
            (inclination_grid.shape[0], 1),
        )
        distances = orthodromic_distance.use_haversine(
            inclination_grid,
            azimuth_grid,
            tiled_viewport_spherical_coords[:, 0],
            tiled_viewport_spherical_coords[:, 1],
        )

        # compute saliency map
        saliency_map = np.exp(
            (-1.0 / (2.0 * np.square(0.1))) * np.square(distances),
        ).reshape(saliency_map_size)
        maps.append(saliency_map)

        # save saliency map to file
        if save_to_file:
            serializer.serialize(  # pragma: no cover
                saliency_maps_dir / f"map_{timestep:.2f}.npy",
                saliency_map,
            )

            image.imwrite(  # pragma: no cover
                saliency_maps_dir / f"map_{timestep:.2f}.jpg",
                (saliency_map * 255).astype(np.uint8),
                color_mode="grayscale",
            )

    return np.asarray(maps, dtype=float)


def extract_sessions(
    list_session_per_video: list[Session],
    saliency_map_size: tuple[int, int],
    save_to_file: bool,
) -> NDArrayFloat:
    r"""Compute the ground-truth saliency maps for a video in multiple sessions.

    $$\text{GT_Saliency}^t_v = \frac{1}{u} \sum_{u}{\text{GT_Saliency}^{t}_{v, u}}$$

    Args:
        list_session_per_video (List[Session]): A list of sessions (same video).
        saliency_map_size (Tuple[int, int]): A tuple of `(height, width)`.
        save_to_file (bool): Allow saving the created saliency maps to `saliency_maps`
            directory in the `session.session_dir` directory.

    Returns:
        NDArrayFloat:
            shape = `(n_samples, saliency_map_height, saliency_map_width)`
    """
    maps_per_session = []

    for session in console.track(list_session_per_video):
        maps = extract_one_session(
            session=session,
            saliency_map_size=saliency_map_size,
            save_to_file=save_to_file,
        )
        maps_per_session.append(maps)

    video_saliency_maps: NDArrayFloat = np.mean(maps_per_session, axis=0)

    return video_saliency_maps
