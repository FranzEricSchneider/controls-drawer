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
        [-0.02, -0.021, 0.0],
        [-0.01, -0.021, 0.0],
        [0.0, -0.021, 0.0],
        [0.01, -0.021, 0.0],
        [0.02, -0.021, 0.0],
        [0.03, -0.021, 0.0],
        [-0.03, -0.014, 0.0],
        [-0.02, -0.014, 0.0],
        [-0.01, -0.014, 0.0],
        [0.0, -0.014, 0.0],
        [0.01, -0.014, 0.0],
        [0.02, -0.014, 0.0],
        [0.03, -0.014, 0.0],
        [-0.03, -0.007, 0.0],
        [-0.02, -0.007, 0.0],
        [-0.01, -0.007, 0.0],
        [0.0, -0.007, 0.0],
        [0.01, -0.007, 0.0],
        [0.02, -0.007, 0.0],
        [0.03, -0.007, 0.0],
        [-0.03, 0.0, 0.0],
        [-0.02, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.03, 0.0, 0.0],
        [-0.03, 0.007, 0.0],
        [-0.02, 0.007, 0.0],
        [-0.01, 0.007, 0.0],
        [0.0, 0.007, 0.0],
        [0.01, 0.007, 0.0],
        [0.02, 0.007, 0.0],
        [0.03, 0.007, 0.0],
        [-0.03, 0.014, 0.0],
        [-0.02, 0.014, 0.0],
        [-0.01, 0.014, 0.0],
        [0.0, 0.014, 0.0],
        [0.01, 0.014, 0.0],
        [0.02, 0.014, 0.0],
        [0.03, 0.014, 0.0],
        [-0.03, 0.021, 0.0],
        [-0.02, 0.021, 0.0],
        [-0.01, 0.021, 0.0],
        [0.0, 0.021, 0.0],
        [0.01, 0.021, 0.0],
        [0.02, 0.021, 0.0],
        [0.03, 0.021, 0.0],
    ])


@pytest.fixture
def axes():
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    return axes


@pytest.fixture
def HT():
    from geometry.planar import Rx, Ry, Rz
    HT = np.eye(4)
    HT[0:3, 0:3] = Rz(np.pi).dot(Rx(7 * np.pi / 6))
    HT[0:3, 3] = np.array([-0.015, 0.08, 0.133])
    # HT[0:3, 0:3] = Rz(np.pi).dot(Rx(14 * np.pi / 12).dot(Ry(np.pi / 4)))
    # HT[0:3, 3] = np.array([0.1, 0.05, 0.09])

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
    pixels = np.array([globalToPixels(point, calibMatrix, HT=HT)
                       for point in rawExteriorPoints])
    return pixels


@pytest.fixture
def rawImFrame(rawPixels, calibMatrix, maxRandom=15):
    from random import random
    from geometry.cameras import pixelsToImFrame
    imFrame = np.array([pixelsToImFrame(pixel + np.array([(random() * maxRandom - maxRandom / 2),
                                                          (random() * maxRandom - maxRandom / 2)]), calibMatrix)
                        for pixel in rawPixels])
    return imFrame


@pytest.fixture
def parameters():
    # Should basically match what's going on in HT
    omega = -np.pi
    phi = -0.01
    kappa = -3.0
    s_14 = 0.0
    s_24 = 0.0
    s_34 = 0.15
    return (omega, phi, kappa, s_14, s_24, s_34)


def testNonLinearFit(rawImFrame, rawExteriorPoints, parameters,
                     HT, rawPixels, calibMatrix, axes):
    from perception.free_parameter_eqs import HTFromParameters
    from perception.free_parameter_eqs import nonLinearLeastSquares
    freeParameters = nonLinearLeastSquares(rawImFrame,
                                           rawExteriorPoints,
                                           parameters,
                                           iterations=60,
                                           plotValues=False)
    foundHT = HTFromParameters(freeParameters)


    import cv2
    from geometry.cameras import globalToPixels

    np.set_printoptions(precision=3, suppress=True)
    print("freeParameters: {}".format(freeParameters))
    print("HT:\n{}".format(HT))
    print("foundHT:\n{}".format(foundHT))
    # assert np.all(np.isclose(HT, foundHT))

    # Uncomment and add axes argument to plot and debug
    from utils.geometry_tools import plotAxes
    plotAxes(axes, np.eye(4), scalar=0.1)
    plotAxes(axes, foundHT, scalar=0.1)
    # Exterior points
    axes.scatter(xs=rawExteriorPoints[:, 0],
                 ys=rawExteriorPoints[:, 1],
                 zs=rawExteriorPoints[:, 2])
    plt.show()

    image = np.ones((480, 640)) * 255
    # Plot raw pixels
    for pixel in rawPixels:
        center = tuple([int(x) for x in pixel])
        cv2.circle(image, center, radius=3, thickness=2, color=0)
    # Recomputed pixels
    invFoundHT = np.linalg.inv(foundHT)
    foundPixels = [globalToPixels(point, calibMatrix, invHT=invFoundHT)
                   for point in rawExteriorPoints]
    for pixel in foundPixels:
        center = tuple([int(x) for x in pixel])
        cv2.circle(image, center, radius=10, thickness=1, color=0)

    cv2.imwrite("test_image.png", image)
    assert False
