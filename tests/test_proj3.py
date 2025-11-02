from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from CurveInterpolation.curve_interpolation import (
    create_coefficients_matrix, create_constants_matrix, solve_for_d_matrix
)
from project3.interpatxy import interpatxy
from tests.conftest import PROJ3_OUTPUT_DIR


class TestCreateCoefficientsMatrix:

    def test_n_3(self):
        """Checking the special case of N = 3"""
        result: np.ndarray = create_coefficients_matrix(3)

        expected: np.ndarray = np.array([
            [4, 1],
            [1, 4],
        ])

        assert np.array_equal(result, expected)

    def test_n_4(self):
        """Checking normal case of N = 4"""
        result: np.ndarray = create_coefficients_matrix(4)

        expected: np.ndarray = np.array([
            [4, 1, 0],
            [1, 4, 1],
            [0, 1, 4],
        ])

        assert np.array_equal(result, expected)

    def test_n_5(self):
        """Checking normal case of N = 5"""
        result: np.ndarray = create_coefficients_matrix(5)

        expected: np.ndarray = np.array([
            [4, 1, 0, 0],
            [1, 4, 1, 0],
            [0, 1, 4, 1],
            [0, 0, 1, 4],
        ])

        assert np.array_equal(result, expected)

    def test_n_6(self):
        """Checking normal case of N = 6"""
        result: np.ndarray = create_coefficients_matrix(6)

        expected: np.ndarray = np.array([
            [4, 1, 0, 0, 0],
            [1, 4, 1, 0, 0],
            [0, 1, 4, 1, 0],
            [0, 0, 1, 4, 1],
            [0, 0, 0, 1, 4],
        ])

        assert np.array_equal(result, expected)


class TestCreateConstantMatrix:

    def test_n_3(self):
        """Check special case of N = 3"""
        points: np.ndarray = np.array([1, 2, 3, 4]).reshape((4, 1))
        result: np.ndarray = create_constants_matrix(points)

        # N = 3
        # d_0 = 2, d_n = 4
        # x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 4
        # 6x_1 - x_0 = 6(2) - 1 = 12 - 1 = 11
        # 6x_2 - x_n = 6(3) - 4 = 18 - 4 = 14
        expected: np.ndarray = np.array([11, 14]).reshape((2, 1))

        assert np.array_equal(result, expected)

    def test_n_4(self):
        """Check special case of N = 3"""
        points: np.ndarray = np.array([1, 2, 3, 4, 5]).reshape((5, 1))
        result: np.ndarray = create_constants_matrix(points)

        # N = 4
        # d_0 = 2, d_n = 4
        # x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 4, x_4 = 5
        # 6x_1 - x_0 = 6(2) - 1 = 12 - 1 = 11
        # 6x_2 = 6(3) = 18
        # 6x_3 - x_4 = 6(4) - 5 = 24 - 5 = 19
        expected: np.ndarray = np.array([11, 18, 19]).reshape((3, 1))

        assert np.array_equal(result, expected)

    def test_n_5(self):
        """Check special case of N = 3"""
        points: np.ndarray = np.array([1, 2, 3, 4, 5, 6]).reshape((6, 1))
        result: np.ndarray = create_constants_matrix(points)

        # N = 5
        # d_0 = 2, d_n = 4
        # x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 4, x_4 = 5, x_5 = 6
        # 6x_1 - x_0 = 6(2) - 1 = 12 - 1 = 11
        # 6x_2 = 6(3) = 18
        # 6x_3 = 6(4) = 24
        # 6x_4 - x_5 = 6(5) - 6 = 30 - 6 = 24
        expected: np.ndarray = np.array([11, 18, 24, 24]).reshape((4, 1))

        assert np.array_equal(result, expected)


class TestSolveForDMatrix:

    def test_n_3(self):
        points: np.ndarray = np.array([1, 2, 3, 4]).reshape((4, 1))

        result: np.ndarray = solve_for_d_matrix(points)

        expected: np.ndarray = np.array([1., 1.33333333, 2., 3., 3.66666667, 4.])

        assert np.allclose(result, expected)

    def test_n_4(self):
        points: np.ndarray = np.array([1, 2, 3, 4, 5]).reshape((5, 1))

        result: np.ndarray = solve_for_d_matrix(points)

        expected: np.ndarray = np.array([1, 1.33333333, 2, 3, 4, 4.66666667, 5])

        assert np.allclose(result, expected)


class TestOutput:

    def test_output(self):
        """
        We are given 1 example of a correct output. Figure 3, at the bottom of the homework 3 pdf,
        has an image of a curve using the following points. We execute our interpatxy method
        with these points and save the image to tests/approval_outputs/proj3/test_img.png
        so we can compare it to the example.
        """

        x_6: np.ndarray = np.array([
            2.7310, 1.3599, 1.1662, 1.9709, 4.7876, 7.0827, 6.2630, 4.2809, 3.9232, 4.9367, 8.3048, 9.0052, 7.6639
        ])
        y_6: np.ndarray = np.array([
            1.1106, 2.4716, 5.3639, 7.3677, 8.3129, 7.6134, 5.3072, 5.0614, 2.8875, 1.0539, 0.8648, 2.5095, 3.7760
        ])

        interpatxy(
            x=x_6,
            y=y_6
        )

        img_output: Path = PROJ3_OUTPUT_DIR / "test_img.png"
        plt.pause(0.5)
        plt.savefig(img_output)
