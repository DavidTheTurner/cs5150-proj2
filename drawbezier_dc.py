import numpy as np

from BezierCurve import BezierCurve

"""
function to draw a Bezier segment
using de Casteljau subdivision
nn = level of subdivision
used by bspline4_dc
also plots the Bezier control polygons if drawb = 1
"""


def drawbezier_dc(B: np.ndarray, nn: int, drawb: bool):
    # nn is the subdivision level

    # === Draw curve here === #
    curve: BezierCurve = BezierCurve(B)
    results: np.ndarray = curve.sub_div(nn).make_list()
    print(results)

    # Plot the curve segment as a random color
    if drawb:
        # Plot bezier points and segments as red +
        pass
    else:
        # Plot bezier points as red +
        pass

    return
