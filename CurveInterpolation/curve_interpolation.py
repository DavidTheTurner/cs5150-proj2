
import numpy as np


def create_coefficients_matrix(n: int) -> np.ndarray:

    coefficients_matrix: np.ndarray
    if n == 3:
        coefficients_matrix = np.array([
            [7/2, 1],
            [1, 7/2],
        ])
    else:
        dim: int = n - 1
        shape: tuple[int, int] = (dim, dim)

        ones: np.ndarray = np.zeros(shape)
        np.fill_diagonal(ones, 1)
        above_diagonal: np.ndarray = np.triu(np.roll(ones, 1, axis=1))
        below_diagonal: np.ndarray = np.tril(np.roll(ones, -1, axis=1))

        coefficients_matrix = np.zeros(shape)
        np.fill_diagonal(coefficients_matrix, 7/2)
        coefficients_matrix = coefficients_matrix + above_diagonal + below_diagonal

    return coefficients_matrix


def create_coefficients_matrix2(n: int) -> np.ndarray:

    coefficients_matrix: np.ndarray
    if n == 3:
        coefficients_matrix = np.array([
            [4, 1],
            [1, 4],
        ])
    else:
        dim: int = n - 1
        shape: tuple[int, int] = (dim, dim)

        ones: np.ndarray = np.zeros(shape)
        np.fill_diagonal(ones, 1)
        above_diagonal: np.ndarray = np.triu(np.roll(ones, 1, axis=1))
        below_diagonal: np.ndarray = np.tril(np.roll(ones, -1, axis=1))

        coefficients_matrix = np.zeros(shape)
        np.fill_diagonal(coefficients_matrix, 4)
        coefficients_matrix = coefficients_matrix + above_diagonal + below_diagonal

    return coefficients_matrix


def create_constants_matrix(points: np.ndarray, d_0: float, d_n: float) -> np.ndarray:
    n: int = points.shape[0] - 1
    assert points.shape[1] == 1, "Points must be columnar."
    assert n >= 3, "Only N >= 3 is supported."

    constants_matrix: np.ndarray = np.zeros((n - 1, 1))
    constants_matrix[0, 0] = 6 * points[1, 0] - (3/2) * d_0

    if n > 3:
        constants_matrix[1:n - 2, 0] = 6 * points[2:n - 1, 0]

    constants_matrix[n - 2, 0] = 6 * points[n - 1, 0] - (3/2) * d_n

    return constants_matrix


def create_derrived_constants_matrix(points: np.ndarray) -> np.ndarray:
    n: int = points.shape[0] - 1
    assert points.shape[1] == 1, "Points must be columnar."
    assert n >= 3, "Only N >= 3 is supported."

    x_0: float = points[0, 0]
    constants_matrix: np.ndarray = np.zeros((n - 1, 1))
    constants_matrix[0, 0] = 6 * points[1, 0] - x_0

    if n > 3:
        constants_matrix[1:n - 2, 0] = 6 * points[2:n - 1, 0]

    x_n: float = points[n, 0]
    constants_matrix[n - 2, 0] = 6 * points[n - 1, 0] - x_n

    return constants_matrix


def solve_for_d_matrix(points: np.ndarray) -> np.ndarray:
    n: int = points.shape[0] - 1

    coefficients_matrix: np.ndarray = create_coefficients_matrix2(n)
    constants_matrix: np.ndarray = create_derrived_constants_matrix(points)

    partial_d_matrix: np.ndarray = np.linalg.solve(coefficients_matrix, constants_matrix)

    d_1: float = partial_d_matrix[0, 0]
    x_0: float = points[0, 0]
    d_n_minus_1: float = partial_d_matrix[-1, 0]
    x_n: float = points[-1, 0]

    d_minus_1: float = x_0
    d_0: float = (2/3) * x_0 + (1/3) * d_1
    d_n: float = (1/3) * d_n_minus_1 + (2/3) * x_n
    d_n_plus_1: float = x_n

    complete_d_matrix: np.ndarray = np.zeros((n + 3, 1))
    complete_d_matrix[0, 0] = d_minus_1
    complete_d_matrix[1, 0] = d_0

    complete_d_matrix[2:-2] = partial_d_matrix

    complete_d_matrix[-2, 0] = d_n
    complete_d_matrix[-1, 0] = d_n_plus_1

    assert complete_d_matrix.shape[0] == n + 3
    return complete_d_matrix
