import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pytest

from geometry.cameras import pixelsToImFrame
from geometry.cameras import imFrameToPixels
from geometry.cameras import globalToPixels
from geometry.cameras import pixelsToGlobalPlane
from geometry.cameras import cropImage
from geometry.planar import Rx, Ry, Rz
from utils.geometry_tools import plotAxes


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


@pytest.fixture
def invCalibMatrix(calibMatrix):
    return np.linalg.inv(calibMatrix)


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
def imFramePoint():
    return np.array([0.0, 0.0, 1.0])


class TestImFrameToPixels():
    def testCallable(self, imFramePoint, calibMatrix):
        pixels = imFrameToPixels(imFramePoint, calibMatrix)

    def testBasicCharacteristics(self, imFramePoint, calibMatrix):
        pixels = imFrameToPixels(imFramePoint, calibMatrix)
        assert pixels.shape == (2,)

    def testCenterNearZero(self, imFramePoint, calibMatrix, centerPoint):
        # Tests that an imframe point near the center of the image gets pixel
        # values of ~c_x/c_y. The noise in the c_x/c_y values means that it's
        # not exact
        pixels = imFrameToPixels(imFramePoint, calibMatrix)
        assert all(abs((pixels - centerPoint) < 5))

    def testErrorCases(self, calibMatrix):
        with pytest.raises(AttributeError):
            pixels = imFrameToPixels("aaa", calibMatrix)
        with pytest.raises(AttributeError):
            pixels = imFrameToPixels([1, 4, 0], calibMatrix)
        with pytest.raises(AttributeError):
            pixels = imFrameToPixels(1.0, calibMatrix)
        with pytest.raises(ValueError):
            pixels = imFrameToPixels(np.array([1, 2]), calibMatrix)
        with pytest.raises(ValueError):
            pixels = imFrameToPixels(np.array([1, 2, 3, 4]), calibMatrix)
        # Should work
        pixels = imFrameToPixels(np.array([1, 2, 3]), calibMatrix)


@pytest.fixture
def globalPoint():
    return np.array([0.1, 0.1, 0.0, 1.0])


@pytest.fixture
def HT():
    # Remember! In an HT the rotation aspect is R.dot(newFrame) = globalFrame,
    # and the transform is in the frame of the globalFrame. This is because it
    # goes v_globalFrame = R.dot(v_newFrame) + T, and so the T has to be in the
    # global coordinate frame
    matrix = np.eye(4)
    matrix[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi))
    matrix[0:3, 3] = np.array([0.1, 0.1, 1.0])
    return matrix


@pytest.fixture
def invHT(HT):
    return np.linalg.inv(HT)


@pytest.fixture
def axes():
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    return axes


class TestGlobalToPixels():
    def testCallable(self, globalPoint, invHT, calibMatrix):
        pixels = globalToPixels(globalPoint, calibMatrix, invHT=invHT)

    def testBasicCharacteristics(self, invHT, calibMatrix):
        zeroPoint = np.array([0, 0, 0, 1])
        zeroHT = np.eye(4)
        zeroHT[2, 3] = -1.0
        pixels = globalToPixels(zeroPoint, calibMatrix, HT=zeroHT)
        assert pixels.shape == (2,)
        assert all(pixels == calibMatrix[0:2, 2])
        zeroHT[2, 3] = -5.0
        pixels = globalToPixels(zeroPoint, calibMatrix, HT=zeroHT)
        assert all(pixels == calibMatrix[0:2, 2])

    def testDirectlyAbove(self, globalPoint, HT, invHT, calibMatrix):
        pixels = globalToPixels(globalPoint, calibMatrix, invHT=invHT)
        assert all(np.isclose(pixels, calibMatrix[0:2, 2]))

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi / 3))
        pixels = globalToPixels(globalPoint, calibMatrix, HT=HT)
        assert all(np.isclose(pixels, calibMatrix[0:2, 2]))

        HT[0:3, 0:3] = Ry(np.pi).dot(Rz(np.pi / 5))
        pixels = globalToPixels(globalPoint, calibMatrix, HT=HT)
        assert all(np.isclose(pixels, calibMatrix[0:2, 2]))

    def testOffsetPoint(self, globalPoint, HT, invHT, calibMatrix):
        # Get the base point
        pixels = globalToPixels(globalPoint, calibMatrix, invHT=invHT)

        offsetPoint = np.array([0.15, 0.15, 0.0, 1.0])
        newPixels = globalToPixels(offsetPoint, calibMatrix, invHT=invHT)
        assert newPixels[0] < pixels[0]
        assert newPixels[1] > pixels[1]

        HT[0:3, 0:3] = Rx(np.pi)
        newPixels = globalToPixels(offsetPoint, calibMatrix, HT=HT)
        assert newPixels[0] > pixels[0]
        assert newPixels[1] < pixels[1]

    def testRotatedViewpoint(self, globalPoint, HT, calibMatrix):
        # Get the base point
        pixels = globalToPixels(globalPoint, calibMatrix, HT=HT)

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Ry(np.pi/8)))
        newPixels = globalToPixels(globalPoint, calibMatrix, HT=HT)
        assert newPixels[0] < pixels[0]
        assert newPixels[1] == pixels[1]

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Ry(np.pi/8).dot(Rx(np.pi/8))))
        newPixels = globalToPixels(globalPoint, calibMatrix, HT=HT)
        assert newPixels[0] < pixels[0]
        assert newPixels[1] > pixels[1]

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Rx(np.pi/8)))
        newPixels = globalToPixels(globalPoint, calibMatrix, HT=HT)
        assert newPixels[0] == pixels[0]
        assert newPixels[1] > pixels[1]

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Rx(-np.pi/8).dot(Ry(-np.pi/8))))
        newPixels = globalToPixels(globalPoint, calibMatrix, HT=HT)
        assert newPixels[0] > pixels[0]
        assert newPixels[1] < pixels[1]

        # Uncomment and add axes argument to plot and debug
        # plotAxes(axes, np.eye(4))
        # plotAxes(axes, invHT)
        # axes.scatter(xs=[globalPoint[0]],
        #              ys=[globalPoint[1]],
        #              zs=[globalPoint[2]])
        # plt.show()
        # assert False

    def testErrorCases(self, invHT, calibMatrix):
        with pytest.raises(AttributeError):
            imFrame = globalToPixels("aaa", calibMatrix, invHT=invHT)
        with pytest.raises(AttributeError):
            imFrame = globalToPixels([1, 4, 0], calibMatrix, invHT=invHT)
        with pytest.raises(AttributeError):
            imFrame = globalToPixels(1.0, calibMatrix, invHT=invHT)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2]), calibMatrix, invHT=invHT)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2, 3, 4, 5]), calibMatrix, invHT=invHT)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2]).reshape((2, 1)), calibMatrix, invHT=invHT)
        with pytest.raises(ValueError):
            imFrame = globalToPixels(np.array([1, 2, 3, 4, 5]).reshape((5, 1)), calibMatrix, invHT=invHT)
        # Should work
        imFrame = globalToPixels(np.array([1, 2, 3]), calibMatrix, invHT=invHT)
        imFrame = globalToPixels(np.array([1, 2, 3, 4]), calibMatrix, invHT=invHT)
        imFrame = globalToPixels(np.array([1, 2, 3]).reshape((3, 1)), calibMatrix, invHT=invHT)
        imFrame = globalToPixels(np.array([1, 2, 3, 4]).reshape((4, 1)), calibMatrix, invHT=invHT)


class TestPixelsToGlobalPlane():
    def testCallable(self, centerPoint, HT, invCalibMatrix):
        point = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)

    def testBasicCharacteristics(self, centerPoint, HT, invCalibMatrix):
        point = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert point.shape == (3,)
        assert np.isclose(point[2], 0.0)
        # The sample HT has a displacement of (0.1, 0.1, 1.0) and the
        # centerPoint points directly at it, but cx/cy errors keep it from
        # being exact
        assert all(abs(point[0:2] - HT[0:2, 3]) < 0.05)

    def testDirectlyAbove(self, centerPoint, HT, invCalibMatrix):
        point = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert all(abs(point[0:2] - HT[0:2, 3]) < 0.05)

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi / 3))
        point = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert all(abs(point[0:2] - HT[0:2, 3]) < 0.05)

        HT[0:3, 0:3] = Ry(np.pi).dot(Rz(np.pi / 5))
        point = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert all(abs(point[0:2] - HT[0:2, 3]) < 0.05)

    def testOffsetPoint(self, centerPoint, HT, invCalibMatrix):
        # Get the base point
        point = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)

        offsetPixels = np.array([0, 0])
        newPoint = pixelsToGlobalPlane(offsetPixels, HT, invCalibMatrix)
        assert newPoint[0] > point[0]
        assert newPoint[1] < point[1]

        HT[0:3, 0:3] = Rx(np.pi)
        newPoint = pixelsToGlobalPlane(offsetPixels, HT, invCalibMatrix)
        assert newPoint[0] < point[0]
        assert newPoint[1] > point[1]

    def testRotatedViewpoint(self, centerPoint, HT, invCalibMatrix):
        # Get the base point
        point = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Ry(np.pi/8)))
        newPoint = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert newPoint[0] < point[0]
        assert abs(newPoint[1] - point[1]) < 0.01

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Ry(-np.pi/8).dot(Rx(np.pi/8))))
        newPoint = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert newPoint[0] > point[0]
        assert newPoint[1] < point[1]

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Rx(np.pi/8)))
        newPoint = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert abs(newPoint[0] - point[0]) < 0.01
        assert newPoint[1] < point[1]

        HT[0:3, 0:3] = Rx(np.pi).dot(Rz(np.pi).dot(Rx(-np.pi/8).dot(Ry(-np.pi/8))))
        newPoint = pixelsToGlobalPlane(centerPoint, HT, invCalibMatrix)
        assert newPoint[0] > point[0]
        assert newPoint[1] > point[1]


@pytest.fixture
def image():
    shape = (480, 640)
    image = np.ones(shape) * 255.0
    for i in xrange(0, shape[0], 10):
        image[i, :] = 0.0
    for j in xrange(0, shape[1], 10):
        image[:, j] = 0.0
    return image


@pytest.fixture
def cropBounds():
    return (100, 200, 150, 275)


class TestCropping():
    def test_callable(self, image, cropBounds, calibMatrix):
        newImage = cropImage(image, cropBounds)
        newImage, newCalibMatrix = cropImage(image, cropBounds, calibMatrix)
        # Check that the image didn't get overwritten
        assert newImage.shape != image.shape

    def test_size(self, image, cropBounds):
        newImage = cropImage(image, cropBounds)
        proposedShape = (cropBounds[1] - cropBounds[0], cropBounds[3] - cropBounds[2])
        print newImage.shape
        print proposedShape
        assert newImage.shape == proposedShape

    def test_match(self, image, cropBounds):
        newImage = cropImage(image, cropBounds)
        assert image[cropBounds[0], cropBounds[2]] == newImage[0, 0]
        assert image[cropBounds[1] - 1, cropBounds[3] - 1] == newImage[-1, -1]

    def test_calib_matrix(self, image, cropBounds, calibMatrix):
        newImage, newCalibMatrix = cropImage(image, cropBounds, calibMatrix)
        assert np.all(calibMatrix[:, 0:2] == newCalibMatrix[:, 0:2])
        assert calibMatrix[0, 2] == newCalibMatrix[0, 2] + cropBounds[2]
        assert calibMatrix[1, 2] == newCalibMatrix[1, 2] + cropBounds[0]
