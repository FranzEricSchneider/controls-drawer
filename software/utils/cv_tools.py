###############################################################################
# Contains cv2 functions and image processing tools
###############################################################################

import cv2
import numpy as np

from geometry.planar import FiniteLine


# NOTE: I haven't really figured out great thresholds and other settings
def calibrationLines(imagePath, threshold1=90, threshold2=100,
                     minLineLength=40):
    image = cv2.imread(imagePath)
    edges = cv2.Canny(image, threshold1, threshold2)
    lines = cv2.HoughLinesP(image=edges,
                            rho=4,
                            theta=np.pi/180,
                            threshold=10,
                            lines=np.array([]),
                            minLineLength=minLineLength,
                            maxLineGap=20)
    finiteLines = [FiniteLine(line[0]) for line in lines]
    return image, finiteLines


def showImage(name, image):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    cv2.waitKey(1)


def writeImage(path, frame):
    cv2.imwrite(path, frame)


def readImage(path):
	return cv2.imread(path)
