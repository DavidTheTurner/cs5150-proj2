
from pathlib import Path
import numpy as np
import pytest


APPROVAL_OUTPUT_DIR: Path = Path("tests/approval_outputs")
PROJ3_OUTPUT_DIR: Path = APPROVAL_OUTPUT_DIR / "proj3"


@pytest.fixture
def d_control_points() -> np.ndarray:
    dx: np.ndarray = np.array([4.2173, 1.5849, 2.1301, 4.7625, 7.7531, 8.3606, 5.4322])
    dy: np.ndarray = np.array([1.8424, 3.2603, 6.0028, 7.6446, 6.3013, 2.5886, 4.0065])
    return np.stack((dx, dy))


@pytest.fixture(autouse=True)
def make_approval_outputs_dir() -> None:
    APPROVAL_OUTPUT_DIR.mkdir(exist_ok=True)
    PROJ3_OUTPUT_DIR.mkdir(exist_ok=True)
