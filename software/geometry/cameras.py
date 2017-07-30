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
        # Remember directionality:
        #   HT.dot(imFrame) = globalFrame
        #   inv(HT).dot(globalFrame) = imFrame
        self.HT = exteriorCalibResults['HT']


def pixelsToImFrame(pixelPoint, calibMatrix):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if pixelPoint.shape != (2,) and pixelPoint.shape != (3,):
        raise ValueError("pixelsToImFrame accepts (2,) or (3,) arrays, not"
                         " {}".format(pixelPoint))
    assert calibMatrix.shape == (3, 3)
    if len(pixelPoint) == 2:
        pixelPoint = np.hstack((pixelPoint, 1))
    return np.linalg.inv(calibMatrix).dot(pixelPoint)


def imFrameToPixels(imFramePoint, calibMatrix):
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if imFramePoint.shape != (3,):
        raise ValueError("imFrameToPixels accepts (3,) arrays, not"
                         " {}".format(imFramePoint))
    assert calibMatrix.shape == (3, 3)
    unscaledPixels = calibMatrix.dot(imFramePoint)
    unitPixels = unscaledPixels / unscaledPixels[2]
    return unitPixels[0:2]


def globalToPixels(point, HT, calibMatrix):
    checkOrthonormal(HT)

    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if point.shape == (3,) or point.shape == (4,):
        point = point.reshape((len(point), 1))
    elif point.shape != (3,1) and point.shape != (4,1):
        raise ValueError("globalToPixels accepts (3,1) or (4,1) arrays, not"
                         " {}".format(point.shape))
    if len(point) == 3:
        point = np.vstack((point, 1))
    imFramePoint = np.linalg.inv(HT).dot(point)
    return imFrameToPixels(imFramePoint[0:3, 0], calibMatrix)


def pixelsToGlobalPlane(pixelPoint, HT, calibMatrix):
    """
    Assuming the global plane is at z=0, project pixel points onto that plane.
    We can't otherwise write pixelsToGlobal because 2D doesn't contain enough
    information
    """
    checkOrthonormal(HT)

    imFramePoint = pixelsToImFrame(pixelPoint, calibMatrix)
    if imFramePoint.shape == (3,1):
        imFramePoint = imFramePoint.reshape((3,))
    imFrameVector = np.hstack((imFramePoint, 0))
    camOrigin = HT[0:3, 3]
    unscaledGlobalVector = HT.dot(imFrameVector)
    # At this point we can calculate the correct scaling by setting the
    # resulting z value equal to 0. If we want to project on a non-(0,0,1)
    # plane then this will get more complicated
    # 0 = camOrigin[2] + k * globalVec[2]
    # k = -camOrigin[2] / globalVec[2]
    scalar = -camOrigin[2] / unscaledGlobalVector[2]
    return camOrigin + scalar * unscaledGlobalVector[0:3]
