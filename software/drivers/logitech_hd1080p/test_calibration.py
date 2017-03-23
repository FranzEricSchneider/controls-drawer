# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# Part 1
import numpy as np
import cv2
import glob


# Checkerboard corner column and row values
CBROW = 9
CBCOL = 6


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...., (6, 5, 0)
objp = np.zeros((CBCOL * CBROW, 3), np.float32)
objp[:, :2] = np.mgrid[0:CBROW, 0:CBCOL].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('square_images/*.jpg')

import ipdb; ipdb.set_trace()
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (CBROW, CBCOL), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (CBROW, CBCOL), corners2, ret)
        # cv2.imshow('img'.format(fname), img)
        # cv2.waitKey(500)
    else:
        print("File {} pattern not found!".format(fname))

cv2.destroyAllWindows()


# Part 2
import ipdb; ipdb.set_trace()
ret, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


# Part 3
h = None
w = None
newCameraMatrix = None
roi = None
for fname in images:
    img = cv2.imread(fname)
    if h is None:
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (w, h), 1, (w, h))

    # Part 4
    # undistort
    dst = cv2.undistort(img, matrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dstCrop = dst[y:y + h, x:x + w]
    cv2.imwrite('{}_calib.png'.format(fname.replace(".jpg", "")), dstCrop)
    cv2.imwrite('{}_calib_nocrop.png'.format(fname.replace(".jpg", "")), dst)


# Part 5
import ipdb; ipdb.set_trace()
tot_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], matrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error

print "mean error: ", tot_error / len(objpoints)
