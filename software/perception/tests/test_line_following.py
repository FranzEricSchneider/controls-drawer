import numpy as np
import pytest

from geometry.planar import Rx, Ry, Rz
from perception.line_following import maskAroundToolframe


@pytest.fixture
def shape():
    return (640, 480)


@pytest.fixture
def centeredHT():
    matrix = np.eye(4)
    matrix[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi))
    matrix[0:3, 3] = np.array([0.0, 0.0, 1.0])
    return matrix


@pytest.fixture
def offcenterHT():
    matrix = np.eye(4)
    matrix[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi))
    matrix[0:3, 3] = np.array([0.1, 0.1, 1.0])
    return matrix


@pytest.fixture
def calibMatrix():
    matrix = np.eye(3)
    # Set the c_x and c_y values for the center of the frame, for a fake
    # (640, 480) camera with added jitter
    matrix[0, 2] = 320
    matrix[1, 2] = 240
    # Set the focal length (I have no idea what a reasonable focal length is)
    matrix[0, 0] = 1000
    matrix[1, 1] = 1000
    return matrix


class TestMaskAroundToolframe():
    def testCallable(self, shape, centeredHT, calibMatrix):
        from cProfile import Profile
        pr = Profile()
        pr.enable()
        mask = maskAroundToolframe(shape, centeredHT, calibMatrix, radius=0.15)
        pr.dump_stats("maskAroundToolframe.runsnake")
        assert False
