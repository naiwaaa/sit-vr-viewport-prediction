#!/usr/bin/env python3
from pathlib import Path

import numpy as np

from viewport_prediction import preprocessors
from viewport_prediction.utils import console
from viewport_prediction.entities import Session


DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"
SALIENCY_MAP_SIZE = (224, 224)


def main() -> None:
    videos_dir = DATA_DIR / "videos"
    video_indices = list(range(1, 6))

    console.rule("has_subtitle group")
    _run_group(DATA_DIR / "has_subtitle", videos_dir, video_indices)


def _run_group(group_dir: Path, videos_dir: Path, video_indices: list[int]) -> None:
    for idx, video_id in enumerate(video_indices):
        console.print(
            f"[{(idx + 1):02}/{len(video_indices):02}] "
            f"Creating ground-truth saliency maps for video {video_id:02}...",
        )

        list_session_per_video = []
        for session_dir in group_dir.glob(f"video_{video_id:02}_user_*"):
            list_session_per_video.append(Session(session_dir))

        saliency_maps = preprocessors.saliency.extract_sessions(
            list_session_per_video,
            SALIENCY_MAP_SIZE,
            save_to_file=True,
        )
        np.save(file=videos_dir / f"video_{video_id:02}_gts.npy", arr=saliency_maps)


if __name__ == "__main__":
    main()
