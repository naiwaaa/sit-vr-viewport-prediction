import numpy as np

from viewport_prediction.helpers import orthodromic_distance
from viewport_prediction.metrics import OrthodromicDistanceMetric


def test_orthodromic_distance_metric() -> None:
    array_1 = np.radians([[60, 0], [39, 0]])
    array_2 = np.radians([[90, 0], [124, -60]])
    distances = orthodromic_distance.use_haversine(
        array_1[:, 0],
        array_1[:, 1],
        array_2[:, 0],
        array_2[:, 1],
    )
    metric = OrthodromicDistanceMetric()
    for pred, target in zip(array_1, array_2):
        metric.update(pred, target)

    result = metric.compute()

    assert result["mean"] == np.mean(distances)
    assert result["std"] == np.std(distances)


def test_reset_orthodromic_distance_metric() -> None:
    metric = OrthodromicDistanceMetric()
    metric.update(np.asarray([1, 2]), np.asarray([2, 3]))

    metric.reset()  # act

    # pylint: disable=protected-access
    assert not metric._all_distances
