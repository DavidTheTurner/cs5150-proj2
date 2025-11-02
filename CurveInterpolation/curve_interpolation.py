
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
