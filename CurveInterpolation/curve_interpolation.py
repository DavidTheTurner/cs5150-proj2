
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
