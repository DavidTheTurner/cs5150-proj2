import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="deboor.log",
    level=logging.INFO
)


def truncate_extra_decimals(array: np.ndarray, decimals: int) -> np.ndarray:
    decimals_array: np.ndarray = array - np.trunc(array)

    decimal_offset: int = 10 ** decimals
    truncated_decimals: np.ndarray = np.trunc(decimals_array * decimal_offset) / decimal_offset

    return np.trunc(array) + truncated_decimals


def segment_c_1(deboor_points: np.ndarray, n: int) -> np.ndarray:
    logger.info(f"Segment C1 w/ Points: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = deboor_points[0]
    b[1] = deboor_points[1]
    b[2] = (1/2) * deboor_points[1] + (1/2) * deboor_points[2]
    b[3] = (1/4) * deboor_points[1] + (7/12) * deboor_points[2] + (1/6) * deboor_points[3]

    return b


def segment_c_2(deboor_points: np.ndarray) -> np.ndarray:
    logger.info(f"Segment C2 w/ Points: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = (1/4) * deboor_points[1] + (7/12) * deboor_points[2] + (1/6) * deboor_points[3]
    b[1] = (2/3) * deboor_points[2] + (1/3) * deboor_points[3]
    b[2] = (1/3) * deboor_points[2] + (2/3) * deboor_points[3]
    b[3] = (1/6) * deboor_points[2] + (4/6) * deboor_points[3] + (1/6) * deboor_points[4]

    return b


def segment_general(deboor_points: np.ndarray, i: int) -> np.ndarray:
    logger.info(f"Segment CGeneral w/ i: {i}\nPoints: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = (1/6) * deboor_points[i - 1] + (4/6) * deboor_points[i] + (1/6) * deboor_points[i + 1]
    b[1] = (2/3) * deboor_points[i] + (1/3) * deboor_points[i + 1]
    b[2] = (1/3) * deboor_points[i] + (2/3) * deboor_points[i + 1]
    b[3] = (1/6) * deboor_points[i] + (4/6) * deboor_points[i + 1] + (1/6) * deboor_points[i + 2]

    return b


def segment_n_sub_3(deboor_points: np.ndarray, n: int) -> np.ndarray:
    logger.info(f"Segment N-3 w/ N: {n}\nPoints: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = (1/6) * deboor_points[n - 4] + (4/6) * deboor_points[n - 3] + (1/6) * deboor_points[n - 2]
    b[1] = (2/3) * deboor_points[n - 3] + (1/3) * deboor_points[n - 2]
    b[2] = (1/3) * deboor_points[n - 3] + (2/3) * deboor_points[n - 2]
    b[3] = (1/6) * deboor_points[n - 3] + (7/12) * deboor_points[n - 2] + (1/4) * deboor_points[n - 1]

    return b


def segment_n_sub_2(deboor_points: np.ndarray, n: int) -> np.ndarray:
    logger.info(f"Segment N-2 w/ N: {n}\nPoints: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = (1/6) * deboor_points[n - 3] + (7 / 12) * deboor_points[n - 2] + (1/4) * deboor_points[n - 1]
    b[1] = (1/2) * deboor_points[n - 2] + (1/2) * deboor_points[n - 1]
    b[2] = deboor_points[n - 1]
    b[3] = deboor_points[n]

    return b


def segment_n_4_c_1(deboor_points: np.ndarray) -> np.ndarray:
    logger.info(f"Segment N = 4, C1 w/ Points: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = deboor_points[0]
    b[1] = deboor_points[1]
    b[2] = (1/2) * deboor_points[1] + (1/2) * deboor_points[2]
    b[3] = (1/4) * deboor_points[1] + (1/2) * deboor_points[2] + (1/4) * deboor_points[3]

    return b


def segment_n_4_n_sub_2(deboor_points: np.ndarray) -> np.ndarray:
    logger.info(f"Segment N = 4, N-2 w/ N: 3\nPoints: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = (1/4) * deboor_points[1] + (1/2) * deboor_points[2] + (1/4) * deboor_points[3]
    b[1] = (1/2) * deboor_points[2] + (1/2) * deboor_points[3]
    b[2] = deboor_points[3]
    b[3] = deboor_points[4]

    return b


def segment_n_5_c_2(deboor_points: np.ndarray) -> np.ndarray:
    logger.info(f"Segment N = 4, C1 w/ Points: {deboor_points}")
    b: np.ndarray = np.zeros(4)
    b[0] = (1/4) * deboor_points[1] + (7/12) * deboor_points[2] + (1/6) * deboor_points[3]
    b[1] = (2/3) * deboor_points[2] + (1/3) * deboor_points[3]
    b[2] = (1/3) * deboor_points[2] + (2/3) * deboor_points[3]
    b[3] = (1/6) * deboor_points[2] + (7/12) * deboor_points[3] + (1/4) * deboor_points[4]

    return b


def get_segment_calculation(points: np.ndarray, segment_index: int, n: int) -> np.ndarray:
    """
    Uses segment_index and n to determine which function to use for calculating
    the points.
    """

    if n == 4:
        if segment_index == 1:
            return segment_n_4_c_1(points)
        else:
            return segment_n_4_n_sub_2(points)

    if n == 5:
        if segment_index == 1:
            return segment_c_1(points, n)
        elif segment_index == 2:
            return segment_n_5_c_2(points)
        else:
            return segment_n_sub_2(points, n)

    if segment_index == 1:
        return segment_c_1(points, n)
    elif segment_index == n - 2:
        return segment_n_sub_2(points, n)
    elif segment_index == 2:
        return segment_c_2(points)
    elif segment_index == n - 3:
        return segment_n_sub_3(points, n)

    return segment_general(points, segment_index)


class DeboorControlPoints:
    n: int
    x_segments: list[np.ndarray]
    y_segments: list[np.ndarray]
    deboor_x_coordinates: np.ndarray
    deboor_y_coordinates: np.ndarray

    def __init__(self, control_points: np.ndarray):
        self.n = control_points.shape[1] - 1
        self.deboor_x_coordinates, self.deboor_y_coordinates = np.unstack(control_points)
        self.x_segments = []
        self.y_segments = []

        # upper bound of range is not inclusive, so n - 1 instead of n - 2
        for segment_index in range(1, self.n - 1):
            x_segment: np.ndarray = get_segment_calculation(
                self.deboor_x_coordinates,
                segment_index,
                self.n
            )
            self.x_segments.append(x_segment)

            y_segment: np.ndarray = get_segment_calculation(
                self.deboor_y_coordinates,
                segment_index,
                self.n
            )
            self.y_segments.append(y_segment)

        self.x_segments = np.vstack(self.x_segments)
        self.y_segments = np.vstack(self.y_segments)

    @property
    def points(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.x_segments,
            self.y_segments,
        )
