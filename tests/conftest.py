
import numpy as np
import pytest


@pytest.fixture
def d_control_points() -> np.ndarray:
    dx: np.ndarray = np.array([4.2173, 1.5849, 2.1301, 4.7625, 7.7531, 8.3606, 5.4322])
    dy: np.ndarray = np.array([1.8424, 3.2603, 6.0028, 7.6446, 6.3013, 2.5886, 4.0065])
    return np.stack((dx, dy))
