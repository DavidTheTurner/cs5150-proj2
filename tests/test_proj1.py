

from pathlib import Path
import numpy as np
import pytest

from BezierCurve import BezierCurve


@pytest.mark.parametrize(
    "depth",
    [(1), (2), (3),]
)
def test_curve_behavior_has_not_changed(d_control_points: np.ndarray, depth: int):

    bezier_curve: BezierCurve = BezierCurve(d_control_points)
    results: np.ndarray = bezier_curve.sub_div(depth).make_list()

    expected_path: Path = Path(f"tests/expected_outputs/expected_depth{depth}.npy")
    expected: np.ndarray = np.load(expected_path)

    assert np.array_equal(results, expected)
