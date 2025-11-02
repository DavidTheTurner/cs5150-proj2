import numpy as np
from CurveInterpolation.curve_interpolation import (
    create_coefficients_matrix, create_coefficients_matrix2, create_constants_matrix, create_derrived_constants_matrix
)


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


class TestCreateCoefficientsMatrix2:

    def test_n_3(self):
        """Checking the special case of N = 3"""
        result: np.ndarray = create_coefficients_matrix2(3)

        expected: np.ndarray = np.array([
            [4, 1],
            [1, 4],
        ])

        assert np.array_equal(result, expected)

    def test_n_4(self):
        """Checking normal case of N = 4"""
        result: np.ndarray = create_coefficients_matrix2(4)

        expected: np.ndarray = np.array([
            [4, 1, 0],
            [1, 4, 1],
            [0, 1, 4],
        ])

        assert np.array_equal(result, expected)

    def test_n_5(self):
        """Checking normal case of N = 5"""
        result: np.ndarray = create_coefficients_matrix2(5)

        expected: np.ndarray = np.array([
            [4, 1, 0, 0],
            [1, 4, 1, 0],
            [0, 1, 4, 1],
            [0, 0, 1, 4],
        ])

        assert np.array_equal(result, expected)

    def test_n_6(self):
        """Checking normal case of N = 6"""
        result: np.ndarray = create_coefficients_matrix2(6)

        expected: np.ndarray = np.array([
            [4, 1, 0, 0, 0],
            [1, 4, 1, 0, 0],
            [0, 1, 4, 1, 0],
            [0, 0, 1, 4, 1],
            [0, 0, 0, 1, 4],
        ])

        assert np.array_equal(result, expected)


class TestCreateConstantsMatrix:

    def test_n_3(self):
        """Check special case of N = 3"""
        points: np.ndarray = np.array([1, 2, 3, 4]).reshape((4, 1))
        result: np.ndarray = create_constants_matrix(points, d_0=2, d_n=4)

        # N = 3
        # d_0 = 2, d_n = 4
        # x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 4
        # 6x_1 - (3/2)d_0 = 6(2) - (3/2)(2) = 12 - 3 = 9
        # 6x_2 - (3/2)d_n = 6(3) - (3/2)(4) = 18 - 6 = 12
        expected: np.ndarray = np.array([9, 12]).reshape((2, 1))

        assert np.array_equal(result, expected)

    def test_n_4(self):
        """Check special case of N = 3"""
        points: np.ndarray = np.array([1, 2, 3, 4, 5]).reshape((5, 1))
        result: np.ndarray = create_constants_matrix(points, d_0=2, d_n=4)

        # N = 4
        # d_0 = 2, d_n = 4
        # x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 4, x_4 = 5
        # 6x_1 - (3/2)d_0 = 6(2) - (3/2)(2) = 12 - 3 = 9
        # 6x_2 = 6(3) = 18
        # 6x_3 - (3/2)d_n = 6(4) - (3/2)(4) = 24 - 6 = 18
        expected: np.ndarray = np.array([9, 18, 18]).reshape((3, 1))

        assert np.array_equal(result, expected)

    def test_n_5(self):
        """Check special case of N = 3"""
        points: np.ndarray = np.array([1, 2, 3, 4, 5, 6]).reshape((6, 1))
        result: np.ndarray = create_constants_matrix(points, d_0=2, d_n=4)

        # N = 5
        # d_0 = 2, d_n = 4
        # x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 4, x_4 = 5, x_5 = 6
        # 6x_1 - (3/2)d_0 = 6(2) - (3/2)(2) = 12 - 3 = 9
        # 6x_2 = 6(3) = 18
        # 6x_3 = 6(4) = 24
        # 6x_4 - (3/2)d_n = 6(5) - (3/2)(4) = 30 - 6 = 24
        expected: np.ndarray = np.array([9, 18, 24, 24]).reshape((4, 1))

        assert np.array_equal(result, expected)


class TestDerrivedCreateConstantMatrix:

    def test_n_3(self):
        """Check special case of N = 3"""
        points: np.ndarray = np.array([1, 2, 3, 4]).reshape((4, 1))
        result: np.ndarray = create_derrived_constants_matrix(points)

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
        result: np.ndarray = create_derrived_constants_matrix(points)

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
        result: np.ndarray = create_derrived_constants_matrix(points)

        # N = 5
        # d_0 = 2, d_n = 4
        # x_0 = 1, x_1 = 2, x_2 = 3, x_3 = 4, x_4 = 5, x_5 = 6
        # 6x_1 - x_0 = 6(2) - 1 = 12 - 1 = 11
        # 6x_2 = 6(3) = 18
        # 6x_3 = 6(4) = 24
        # 6x_4 - x_5 = 6(5) - 6 = 30 - 6 = 24
        expected: np.ndarray = np.array([11, 18, 24, 24]).reshape((4, 1))

        assert np.array_equal(result, expected)
