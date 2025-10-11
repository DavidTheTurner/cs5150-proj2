
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pytest

from BezierCurve import BezierCurve, DeboorControlPoints
from bspline2b import bspline2b


@pytest.fixture
def d_control_points() -> np.ndarray:
    dx: np.ndarray = np.array([4.2173, 1.5849, 2.1301, 4.7625, 7.7531, 8.3606, 5.4322])
    dy: np.ndarray = np.array([1.8424, 3.2603, 6.0028, 7.6446, 6.3013, 2.5886, 4.0065])
    return np.stack((dx, dy))


@pytest.mark.parametrize(
    "depth",
    [(1), (2), (3),]
)
def test_behavior_has_not_changed(d_control_points: np.ndarray, depth: int):

    bezier_curve: BezierCurve = BezierCurve(d_control_points)
    results: np.ndarray = bezier_curve.sub_div(depth).make_list()

    expected_path: Path = Path(f"expected_outputs/expected_depth{depth}.npy")
    expected: np.ndarray = np.load(expected_path)

    assert np.array_equal(results, expected)


def test_example(d_control_points: np.ndarray):
    p = Path("./output/part2")
    p.mkdir(exist_ok=True, parents=True)
    bspline2b(
        d_control_points[0],
        d_control_points[1],
        6,
        6,
        True
    )  # TODO: fix error checking here
    imname = os.path.join("output", "part2", "showb0_1.png")
    plt.pause(0.5)
    plt.savefig(imname)
    plt.close('all')


def test_it_is_actually_correct(d_control_points: np.ndarray):

    control_points: DeboorControlPoints = DeboorControlPoints(d_control_points)
    bx, by = control_points.points

    expected_x: np.ndarray = np.array(
        [
            [4.2173, 1.5849, 1.8575, 2.4325],
            [2.4325, 3.0075, 3.8850, 4.8222],
            [4.8222, 5.7593, 6.7562, 7.4065],
            [7.4065, 8.0569, 8.3606, 5.4322],
        ]
    )

    expected_y: np.ndarray = np.array(
        [
            [1.8424, 3.2603, 4.6315, 5.5908],
            [5.5908, 6.4401, 7.0973, 7.1471],
            [7.1471, 7.1968, 6.7491, 5.5970],
            [5.5970, 4.4450, 2.5886, 4.0065],
        ]
    )

    assert np.allclose(bx, expected_x)
    assert np.allclose(by, expected_y)
