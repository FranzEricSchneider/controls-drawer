import numpy as np
import pytest

from geometry.cameras import pixelsToImFrame
from geometry.cameras import globalToPixels
from geometry.planar import Rx, Ry, Rz


@pytest.fixture
def centerPoint():
    return np.array([320, 240])


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


class TestPixelsToImFrame():
    def testCallable(self, centerPoint, calibMatrix):
        imFrame = pixelsToImFrame(centerPoint, calibMatrix)

    def testBasicCharacteristics(self, centerPoint, calibMatrix):
        imFrame = pixelsToImFrame(centerPoint, calibMatrix)
        assert imFrame.shape == (3,)
        assert imFrame[2] == 1

    def testCenterNearZero(self, centerPoint, calibMatrix):
        # Tests that a point near the center of the image gets an image frame
        # value of ~0. The noise in the c_x/c_y values means that it's not
        # exactly 0
        imFrame = pixelsToImFrame(centerPoint, calibMatrix)
        assert all(abs(imFrame[0:2] < 0.005))

    def testErrorCases(self, calibMatrix):
        with pytest.raises(AttributeError):
            imFrame = pixelsToImFrame("aaa", calibMatrix)
        with pytest.raises(AttributeError):
            imFrame = pixelsToImFrame([1, 4, 0], calibMatrix)
        with pytest.raises(AttributeError):
            imFrame = pixelsToImFrame(1.0, calibMatrix)
        with pytest.raises(ValueError):
            imFrame = pixelsToImFrame(np.array([1]), calibMatrix)
        with pytest.raises(ValueError):
            imFrame = pixelsToImFrame(np.array([1, 2, 3, 4]), calibMatrix)
        # Should work
        imFrame = pixelsToImFrame(np.array([1, 2]), calibMatrix)
        imFrame = pixelsToImFrame(np.array([1, 2, 3]), calibMatrix)


@pytest.fixture
def globalPoint():
    return np.array([0.1, 0.1, 0.0, 1.0])


@pytest.fixture
def HT():
    # Remember! In an HT the rotation aspect is R.dot(globalFrame) = newFrame,
    # but the transform is in the frame of the newFrame. This is because it
    # goes v_newFrame = R.dot(v_globalFrame) + T, and so the T has to be in the
    # new coordinate frame
    matrix = np.eye(4)
    matrix[0:3, 0:3] = Rz(np.pi).dot(Rx(np.pi))
    matrix[0:3, 3] = np.array([0.1, -0.1, 1.0])
    return matrix


class TestGlobalToPixels():
    def testCallable(self, globalPoint, HT, calibMatrix):
        pixels = globalToPixels(globalPoint, HT, calibMatrix)

    def testBasicCharacteristics(self, HT, calibMatrix):
        zeroPoint = np.array([0, 0, 0, 1])
        zeroHT = np.eye(4)
        zeroHT[2, 3] = -1.0
        pixels = globalToPixels(zeroPoint, zeroHT, calibMatrix)
        assert pixels.shape == (2,)
        assert all(pixels == calibMatrix[0:2, 2])
        zeroHT[2, 3] = -5.0
        pixels = globalToPixels(zeroPoint, zeroHT, calibMatrix)
        assert all(pixels == calibMatrix[0:2, 2])
        zeroHT[0:3, 0:3] = Rz(np.pi / 7)

    def testPointPlacement(self, globalPoint, HT, calibMatrix):
        pixels = globalToPixels(globalPoint, HT, calibMatrix)
        assert all(pixels == calibMatrix[0:2, 2])
        offsetPoint = np.array([0.15, 0.15, 0.0, 1.0])
        newPixels = globalToPixels(offsetPoint, HT, calibMatrix)
        assert newPixels[0] < pixels[0]
        assert newPixels[1] > pixels[1]
        HT[0:3, 0:3] = Rx(np.pi)
        newPixels = globalToPixels(offsetPoint, HT, calibMatrix)
        assert newPixels[0] > pixels[0]
        assert newPixels[1] < pixels[1]
        HT[0:3, 0:3] = Ry(np.pi/8).dot(Rz(np.pi).dot(Rx(np.pi)))
        newPixels = globalToPixels(globalPoint, HT, calibMatrix)
        assert newPixels[0] > pixels[0]
        assert newPixels[1] == pixels[1]

    def testErrorCases(self, HT, calibMatrix):
        with pytest.raises(AttributeError):
            imFrame = globalToPixels("aaa", HT, calibMatrix)
        with pytest.raises(AttributeError):
            imFrame = globalToPixels([1, 4, 0], HT, calibMatrix)
        with pytest.raises(AttributeError):
            imFrame = globalToPixels(1.0, HT, calibMatrix)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2]), HT, calibMatrix)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2, 3, 4, 5]), HT, calibMatrix)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2]).reshape((2, 1)), HT, calibMatrix)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2, 3, 4, 5]).reshape((5, 1)), HT, calibMatrix)
        # Should work
        imFrame = globalToPixels(np.array([1, 2, 3]), HT, calibMatrix)
        imFrame = globalToPixels(np.array([1, 2, 3, 4]), HT, calibMatrix)
        imFrame = globalToPixels(np.array([1, 2, 3]).reshape((3, 1)), HT, calibMatrix)
        imFrame = globalToPixels(np.array([1, 2, 3, 4]).reshape((4, 1)), HT, calibMatrix)
