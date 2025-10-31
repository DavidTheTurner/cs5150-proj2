
import numpy as np

from DeboorPoints import DeboorControlPoints


def test_it_is_actually_correct(d_control_points: np.ndarray):

    control_points: DeboorControlPoints = DeboorControlPoints(d_control_points)
    bx, by = control_points.points

    expected_x: np.ndarray = np.array(
        [
            [4.2173, 1.5849, 1.8575, 2.43253333],
            [2.43253333, 3.00756667, 3.88503333, 4.8222],
            [4.8222, 5.75936667, 6.75623333, 7.40654167],
            [7.40654167, 8.05685, 8.3606, 5.4322],
        ]
    )

    expected_y: np.ndarray = np.array(
        [
            [1.8424, 3.2603, 4.63155, 5.59080833],
            [5.59080833, 6.55006667, 7.09733333, 7.14708333],
            [7.14708333, 7.19683333, 6.74906667, 5.59700833],
            [5.59700833, 4.44495, 2.5886, 4.0065],
        ]
    )

    assert np.allclose(bx, expected_x)
    assert np.allclose(by, expected_y)
