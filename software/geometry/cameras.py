import numpy as np
import pickle

from utils.geometry_tools import checkOrthonormal
from utils import navigation


class Camera():
    """
    General purpose camera class that will store geometrical information about
    the camera for easy reference
    """
    def __init__(self):
        intrinsicCalibResults = pickle.load(
            open(navigation.getLatestIntrinsicCalibration(), "rb"))
        exteriorCalibResults = pickle.load(
            open(navigation.getLatestExteriorCalibration(), "rb"))
        self.calibMatrix = intrinsicCalibResults['matrix']
        self.invCalibMatrix = np.linalg.inv(self.calibMatrix)
        # Remember directionality:
        #   HT.dot(imFrame) = globalFrame
        #   inv(HT).dot(globalFrame) = imFrame
        self.HT = exteriorCalibResults['HT']
        self.invHT = np.linalg.inv(self.HT)


def pixelsToImFrame(pixelPoint, calibMatrix=None, invCalibMatrix=None):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if invCalibMatrix is None:
        invCalibMatrix = np.linalg.inv(calibMatrix)
    if pixelPoint.shape[0] != 2 and pixelPoint.shape[0] != 3:
        raise ValueError("pixelsToImFrame accepts (2,:) or (3,:) arrays, not"
                         " {}".format(pixelPoint.shape))
    if pixelPoint.shape[0] == 2 and len(pixelPoint.shape) == 1:
        pixelPoint = np.hstack((pixelPoint, 1))
    elif pixelPoint.shape[0] == 2:
        pixelPoint = np.vstack((pixelPoint, np.ones(1, pixelPoint.shape[1])))
    return invCalibMatrix.dot(pixelPoint)


def imFrameToPixels(imFramePoint, calibMatrix):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if imFramePoint.shape != (3,):
        raise ValueError("imFrameToPixels accepts (3,) arrays, not"
                         " {}".format(imFramePoint))
    unscaledPixels = calibMatrix.dot(imFramePoint)
    unitPixels = unscaledPixels / unscaledPixels[2]
    return unitPixels[0:2]


def globalToPixels(point, calibMatrix, HT=None, invHT=None):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if invHT is None:
        invHT = np.linalg.inv(HT)
    if point.shape == (3,) or point.shape == (4,):
        point = point.reshape((len(point), 1))
    elif point.shape != (3,1) and point.shape != (4,1):
        raise ValueError("globalToPixels accepts (3,1) or (4,1) arrays, not"
                         " {}".format(point.shape))
    if len(point) == 3:
        point = np.vstack((point, 1))
    imFramePoint = invHT.dot(point)
    return imFrameToPixels(imFramePoint[0:3, 0], calibMatrix)


def pixelsToGlobalPlane(pixelPoint, HT, invCalibMatrix):
    """
    Assuming the global plane is at z=0, project pixel points onto that plane.
    We can't otherwise write pixelsToGlobal because 2D doesn't contain enough
    information
    Accepts either a single point shape=(3,), or a set of points, shape=(3,n)
    """
    imFramePoints = pixelsToImFrame(pixelPoint, invCalibMatrix=invCalibMatrix)
    camOrigin = HT[0:3, 3]
    # By only using the rotation matrix we make this a vector, not a point
    unscaledGlobalVectors = HT[0:3, 0:3].dot(imFramePoints)
    # At this point we can calculate the correct scaling by setting the
    # resulting z value equal to 0. If we want to project on a non-(0,0,1)
    # plane then this will get more complicated
    # 0 = camOrigin[2] + k * globalVec[2]
    # k = -camOrigin[2] / globalVec[2]
    if len(unscaledGlobalVectors.shape) == 1:
        scalar = -camOrigin[2] / unscaledGlobalVectors[2]
    else:
        scalar = -camOrigin[2] / unscaledGlobalVectors[2, :]
        camOrigin = camOrigin.reshape((3, 1))
    return camOrigin + scalar * unscaledGlobalVectors


def getMaskBounds(mask, pixBuffer=5):
    """
    Takes a mask and returns the (iMin, iMax, jMin, jMax) bounds that offset
    that mask by a `pixBuffer` pixels on either side
    """
    assert len(mask.shape) == 2
    assert mask.dtype == np.bool
    indices = np.argwhere(mask)
    iMin = max(indices[:, 0].min() - pixBuffer, 0)
    iMax = min(indices[:, 0].max() + pixBuffer, mask.shape[0] - 1)
    jMin = max(indices[:, 1].min() - pixBuffer, 0)
    jMax = min(indices[:, 1].max() + pixBuffer, mask.shape[1] - 1)
    return (iMin, iMax, jMin, jMax)


def cropImage(image, bounds, calibMatrix=None):
    """
    Crops the image and adjusts the calibMatrix for that image appropriately.
    The crop bounds will uniquely alter the calibMatrix so it will need to
    stay bound with the image.

    Bounds should be given in (iMin, iMax, jMin, jMax) form. Note that i
    corresponds to Y and j to X. The max values are not inclusive

    How cropping affects the intrinsic camera matrix
    Clear but not detailed: https://stackoverflow.com/questions/22437737/opencv-camera-calibration-of-an-image-crop-roi-submatrix
    Detailed but not clear: https://www.quora.com/In-camera-calibration-how-does-the-intrinsic-matrix-change-after-center-cropped-into-a-lower-resolution-image
    """
    iMin, iMax, jMin, jMax = bounds
    # Do input checks
    assert all([m >= 0 for m in [iMin, jMin, iMax, jMax]])
    assert all([m <= (image.shape[0] - 1) for m in [iMin, iMax]])
    assert all([m <= (image.shape[1] - 1) for m in [jMin, jMax]])
    assert iMax > iMin
    assert jMax > jMin
    # Crop down the array
    image = image[iMin:iMax, jMin:jMax]
    if calibMatrix is None:
        return image

    newCalibMatrix = calibMatrix.copy()
    newCalibMatrix[0, 2] -= jMin
    newCalibMatrix[1, 2] -= iMin
    return image, newCalibMatrix


# How resizing affects the intrinsic camera matrix
# https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
