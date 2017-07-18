# So this will be fairly involved, and may end up not being much of an
# automated test, but I want to make a simulated setup for convergence and
# tests/ seems like a reasonable place to do it

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pytest


@pytest.fixture
def rawExteriorPoints():
    # Defined in meters, in the tool frame
    return np.array([
        [-0.03, -0.021, 0.0],
        [-0.02, -0.014, 0.0],
        [-0.01, -0.007, 0.0],
        [0.0, 0.0, 0.0],
        [0.01, 0.007, 0.0],
        [0.02, 0.014, 0.0],
        [0.03, 0.021, 0.0],
        [-0.03, 0.021, 0.0],
        [-0.02, 0.014, 0.0],
        [-0.01, 0.007, 0.0],
        [0.01, -0.007, 0.0],
        [0.02, -0.014, 0.0],
        [0.03, -0.021, 0.0],
    ])


@pytest.fixture
def axes():
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    return axes


@pytest.fixture
def HT():
    from geometry.planar import Rx, Rz
    HT = np.eye(4)
    HT[0:3, 0:3] = Rz(np.pi).dot(Rx(np.pi))
    HT[0:3, 3] = np.array([0, 0, 0.1])
    # HT[0:3, 0:3] = Rz(np.pi).dot(Rx(13 * np.pi / 12))
    # HT[0:3, 3] = np.array([0, 0.03, 0.1])

    # # Uncomment and add axes argument to plot and debug
    # from utils.geometry_tools import plotAxes
    # plotAxes(axes, np.eye(4))
    # plotAxes(axes, HT)
    # plt.show()
    # assert False
    return HT


@pytest.fixture
def calibMatrix():
    matrix = np.eye(3)
    # Set the c_x and c_y values for the center of the frame, for a fake
    # (640, 480) camera with added jitter
    matrix[0, 2] = 322
    matrix[1, 2] = 239
    # Set the focal length (I have no idea what a reasonable focal length is)
    matrix[0, 0] = 1000
    matrix[1, 1] = 1000
    return matrix


@pytest.fixture
def rawPixels(rawExteriorPoints, HT, calibMatrix):
    from geometry.cameras import globalToPixels
    pixels = np.array([globalToPixels(point, HT, calibMatrix)
                       for point in rawExteriorPoints])
    return pixels


@pytest.fixture
def rawImFrame(rawPixels, calibMatrix):
    from geometry.cameras import pixelsToImFrame
    imFrame = np.array([pixelsToImFrame(pixel, calibMatrix)
                        for pixel in rawPixels])
    return imFrame


def testNonLinearFit(rawImFrame, rawExteriorPoints, HT):
    from perception.free_parameter_eqs import HTFromParameters
    from perception.free_parameter_eqs import nonLinearLeastSquares
    freeParameters = nonLinearLeastSquares(rawImFrame, rawExteriorPoints, iterations=60, plotValues=True)
    print freeParameters
    assert False
