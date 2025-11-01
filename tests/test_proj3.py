import numpy as np
from CurveInterpolation.curve_interpolation import create_coefficients_matrix


class TestCreateCoefficientsMatrix:

    def test_n_3(self):
        """Checking the special case of N = 3"""
        result: np.ndarray = create_coefficients_matrix(3)

        expected: np.ndarray = np.array([
            [7/2, 1],
            [1, 7/2],
        ])

        assert np.array_equal(result, expected)

    def test_n_4(self):
        """Checking normal case of N = 4"""
        result: np.ndarray = create_coefficients_matrix(4)

        expected: np.ndarray = np.array([
            [7/2, 1, 0],
            [1, 7/2, 1],
            [0, 1, 7/2],
        ])

        assert np.array_equal(result, expected)

    def test_n_5(self):
        """Checking normal case of N = 5"""
        result: np.ndarray = create_coefficients_matrix(5)

        expected: np.ndarray = np.array([
            [7/2, 1, 0, 0],
            [1, 7/2, 1, 0],
            [0, 1, 7/2, 1],
            [0, 0, 1, 7/2],
        ])

        assert np.array_equal(result, expected)

    def test_n_6(self):
        """Checking normal case of N = 6"""
        result: np.ndarray = create_coefficients_matrix(6)

        expected: np.ndarray = np.array([
            [7/2, 1, 0, 0, 0],
            [1, 7/2, 1, 0, 0],
            [0, 1, 7/2, 1, 0],
            [0, 0, 1, 7/2, 1],
            [0, 0, 0, 1, 7/2],
        ])

        assert np.array_equal(result, expected)
