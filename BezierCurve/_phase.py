import numpy as np


class Phase:
    dimension: int
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray

    def __init__(self, control_points: np.ndarray):
        number_of_control_points: int = control_points.shape[1]

        self.x_coordinates: np.ndarray = np.zeros(
            shape=(number_of_control_points, number_of_control_points),
        )
        self.x_coordinates[0] = control_points[0]

        self.y_coordinates: np.ndarray = np.zeros(
            shape=(number_of_control_points, number_of_control_points)
        )
        self.y_coordinates[0] = control_points[1]

        self.dimension: int = number_of_control_points
        self.interpolate()

    def interpolate(self) -> "Phase":

        for row in range(0, self.dimension - 1):
            x_row: np.ndarray = self.x_coordinates[row]
            y_row: np.ndarray = self.y_coordinates[row]

            # offset is just: n-1, n-2, n-3, etc.
            offset: int = self.dimension - row - 1

            # 0.5b_i + 0.5b_{i + 1}
            self.x_coordinates[row + 1, 0:offset] = (
                0.5 * x_row[0:offset] + 0.5 * np.roll(x_row, -1)[0:offset]
            )
            self.y_coordinates[row + 1, 0:offset] = (
                0.5 * y_row[0:offset] + 0.5 * np.roll(y_row, -1)[0:offset]
            )

        return self

    @property
    def ud(self) -> np.ndarray:
        ud_x: np.ndarray = self.x_coordinates[:, 0]
        ud_y: np.ndarray = self.y_coordinates[:, 0]
        return np.stack((ud_x, ud_y))

    @property
    def ld(self) -> np.ndarray:
        ld_x: np.ndarray = np.flipud(self.x_coordinates).diagonal()
        ld_y: np.ndarray = np.flipud(self.y_coordinates).diagonal()
        return np.stack((ld_x, ld_y))

    @property
    def ud_and_ld(self) -> np.ndarray:
        ud_and_ld: np.ndarray = np.concat((self.ud, self.ld[:, 1:]), axis=1)
        return ud_and_ld
