
from itertools import chain
import numpy as np
from ._phase import Phase


class BezierCurve:
    control_points: np.ndarray
    phases: list[list[np.ndarray]]

    def __init__(self, control_points: np.ndarray):
        self.control_points = control_points

        phase_0: Phase = Phase(self.control_points)
        self.phases = [[phase_0]]

    def sub_div(self, iterations: int) -> "BezierCurve":
        for _ in range(1, iterations):
            last_iteration: list[Phase] = self.phases[-1]

            # Split ud and ld so the phases stay the same size as the input
            # cpoly
            new_iteration: list[Phase] = list(chain.from_iterable([
                (Phase(lpoly.ud), Phase(lpoly.ld))
                for lpoly in last_iteration
            ]))

            self.phases.append(new_iteration)

        return self

    def get_output(self) -> np.ndarray:
        last_iteration: list[Phase] = self.phases[-1]
        results: np.ndarray = np.array(
            [i for lpoly in last_iteration for i in [lpoly.ud, lpoly.ld]]
        )
        return results

    def make_list(self) -> np.ndarray:
        lpoly: np.ndarray = self.get_output()
        new_poly: list = [poly for poly in lpoly]
        new_poly[1:] = [p[:, 1:] for p in new_poly[1:]]
        results: np.ndarray = np.concat(new_poly, axis=1)
        print(results)
        return results
