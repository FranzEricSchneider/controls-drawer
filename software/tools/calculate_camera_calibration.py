###############################################################################
# Runs the functions that make camera calibration happen
###############################################################################

import argparse
import cv2
from glob import glob
import json
import numpy as np
from os import path
import pickle

from geometry import planar
from free_parameter_tools import matrix_row, function1, function2, HT_from_parameters
from utils import cv_tools
from utils import geometry_tools
from utils import navigation
from utils import ui


def cameraCalibration(args):
    # Finds the folder with the calibration images
    if args.folder_name is None:
        # If you aren't given a folder, check all the calibration image folders
        # in the results directory
        resultsDirectory = navigation.resultsDir()
        imageDirectory = path.join(resultsDirectory, "calibration_images")
        searchString = path.join(imageDirectory, "frames*")
        # Sort the directories - because the names should be identical except
        # for the utime (folder name should be 'frames_utime') this will put
        # the most recently made folders last
        possibleDirectories = sorted(glob(searchString))
        # Check that there are valid folders
        if len(possibleDirectories) < 1:
            raise RuntimeError("There are no calibration directories")
        # Set the oldest folder as the designated one
        folder = possibleDirectories[-1]
    elif args.folder_name.startswith("/home"):
        # Check and see if the path is global
        folder = navigation.validDir(args.folder_name)
    else:
        # Check if the path is in results/calibration_images/
        resultsDirectory = navigation.resultsDir()
        imageDirectory = path.join(resultsDirectory, "calibration_images")
        folder = navigation.validDir(path.join(imageDirectory,
                                               args.folder_name))

    # Find the images that we want to run this on
    imagePaths = glob(path.join(folder, "*.png"))
    if len(imagePaths) < 1:
        raise RuntimeError("No images in folder {}".format(folder))

    for imagePath in imagePaths:
        # Runs the pentagon finding code IF there is no metadata file OR if the
        # args decree it
        metadata = getMetadata(imagePath)
        imageName = path.basename(imagePath)
        if not path.isfile(metadata) or args.run_pentagon_finding:
            print("Finding pentagon in {}".format(imageName))

            # Get the image and all available lines
            image, finiteLines = cv_tools.calibrationLines(imagePath)
            # Get the starting indices for the lines
            indices = range(len(finiteLines))

            # Remove lines the user doesn't like
            lineRemover = ui.AskAboutLinesToRemove(image, finiteLines, indices)
            lineRemover.processImage()
            indices = lineRemover.getIndices()
            finiteLines = lineRemover.getLines()

            # Merge lines the user wants to merge
            lineMerger = ui.AskAboutLinesToMerge(image, finiteLines, indices)
            lineMerger.processImage()
            indices = lineMerger.getIndices()
            finiteLines = lineMerger.getLines()

            # Have the user identify which pentagon sides go with which lines
            pentagon = ui.AskAboutPentagonLines(image, finiteLines, indices)
            pentagon.processImage()
            indicesBySide = pentagon.getIndicesBySide()

            # Calculate the pentagon points in pixel coordinates and write to
            # the metadata file
            # First calculate the midlines of each pair of edge lines
            midLines = [
                finiteLines[indicesBySide[i][0]].average(
                    finiteLines[indicesBySide[i][1]])
                for i in xrange(5)
            ]
            midInfiniteLines = [planar.InfiniteLine(fLine=line)
                                for line in midLines]
            # Then get the intersection points of those midlines to find the
            # corners of the pentagon
            midIndices = np.array(range(len(midLines)))
            vertices = [midInfiniteLines[i].intersection(midInfiniteLines[j])
                        for i, j in zip(midIndices, np.roll(midIndices, 1))]

            # Write the lines/vertices data to file
            data = {"vertices": [list(vertex) for vertex in vertices]}
            with open(metadata, "w") as outfile:
                json.dump(data, outfile)
            print("Wrote vertices to {}".format(metadata))

            # Plot the final pentagon if desired
            if args.plot_results:
                finalImage = image.copy()
                for line in midInfiniteLines:
                    line.onImage(finalImage, thickness=1)
                for line in midLines:
                    line.onImage(finalImage, color=(0, 255, 0), thickness=2)
                for vertex in vertices:
                    center = tuple([int(x) for x in vertex])
                    cv2.circle(finalImage, center, radius=6, thickness=2,
                               color=(204, 255, 0))
                cv_tools.showImage(metadata, finalImage)

        elif args.plot_results:
            print("Displaying pre-found pentagon {}".format(imageName))

            # Read in image and point data
            image = cv_tools.readImage(imagePath)
            with open(metadata, "r") as infile:
                data = json.load(infile)

            # Plot and show those vertices
            for vertex in data["vertices"]:
                center = tuple([int(x) for x in vertex])
                cv2.circle(image, center, radius=6, thickness=2,
                           color=(204, 255, 0))
            cv_tools.showImage(metadata, image)

        else:
            print("Found file containing point pairs for {}".format(imageName))

    # Time to take the pixel vertex values, real world displacement values, and
    # combine them into a transformation from tooltip to camera. Check out
    # https://www.sharelatex.com/project/586949e817ccee00403fbc56 for the math
    # behind this part
    vertices, exteriorPts = getCalibPoints(imagePaths)
    # Get the interior camera calibration data to get a number for focal length
    calibrationResults = pickle.load(
        open(navigation.getLatestIntrinsicCalibration(), "rb"))
    focalX = calibrationResults['matrix'][0, 0]
    focalY = calibrationResults['matrix'][1, 1]
    # Take the average of x and y, they are already almost equal
    f = (focalX + focalY) / 2.0

    # Calculate the 6 free parameters that make up a homogeneous transform,
    # three Euler angles and three translation distances
    freeParameters = nonLinearLeastSquares(f, vertices, exteriorPts)
    HT = HT_from_parameters(freeParameters)
    geometry_tools.checkOrthonormal(HT)


def getCalibPoints(imagePaths):
    """
    Assuming the pentagon points have been extracted from the given images,
    returns the point locations in pixel and exterior coordinates

    Inputs:
        imagePaths: A list of paths to images with extracted pentagons

    Outputs:
        vertices: An nx2 array of all pentagon points in pixel coordinates
        exteriorPts: An nx3 array of all pentagon points in exterior (tooltip)
                     coordinates, the reference frame that we want to relate to
                     the camera
    """

    vertices = []
    exteriorPts = []

    for imagePath in imagePaths:
        metadata = getMetadata(imagePath)
        imageName = path.basename(imagePath)

        # Get vertices from the metadata
        with open(metadata, "r") as infile:
            data = json.load(infile)
        vertices.extend(data["vertices"])

        # Parse tooltip data out of the image name
        imageNameSplit = imageName.lower().split("_")
        sideLength = None
        x = None
        y = None
        for part in imageNameSplit:
            if part.startswith("sl"):
                # Length of pentagon side, in meters
                sideLength = float(part.replace("sl", "")) / 1000
            if part.startswith("x"):
                # X travel from 1st point to image point, in meters
                x = float(part.replace("x", "")) / 1000
            if part.startswith("y"):
                # Y travel from 1st point to image point, in meters
                y = float(part.replace("y", "")) / 1000
        if sideLength is None or x is None or y is None:
            raise ValueError("Image {} lacks data".format(imageName))

        # Solve for the pentagon points in the exterior frame
        pentagonVectors = planar.polygonVectors(5)
        from random import random
        pointsFromCamera = pentagonVectors * sideLength - np.array([x, y, 0.0])
        for line in pointsFromCamera:
            line[-1] += (random() - 0.5) * 0.002
        exteriorPts.extend(list(pointsFromCamera))

    return (vertices, exteriorPts)


def getMetadata(imagePath):
    """
    Takes an image name and returns the name of the file that contains the
    pentagon point data
    """
    imageName = path.basename(imagePath)
    fileName = "pentagon_" + imageName.replace("png", "json")
    return imagePath.replace(imageName, fileName)


def nonLinearLeastSquares(f, vertices, exteriorPts, plotValues=False):
    # Method and many of the more meaningless names taken from here:
    # http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html

    # Choose initial values for the free parameters
    phi = 3.0
    omega = 0.1
    kappa = 0.05
    s_14 = 0.02
    s_24 = 0.01
    s_34 = 0.04
    # Track the parameters over time in a matrix, use the latest values to
    # calculate each consecutive step
    freeParameters = np.array([phi, omega, kappa, s_14, s_24, s_34]).reshape(6, 1)

    # TODO: Make this for loop a combination of delta resolution and maximum
    #       iterations
    for i in range(10):
        # Loop through every measurement point
        residuals = None
        AMatrix = None
        for i in xrange(len(vertices)):
            x_1 = exteriorPts[i][0]
            x_2 = exteriorPts[i][1]
            x_3 = exteriorPts[i][2]

            # y1 and y2 are the "measured output" variables
            y1 = vertices[i][0] / f
            y2 = vertices[i][1] / f

            # Calculate current residuals
            newResiduals = np.array([
                [y1 - function1(freeParameters[:, -1], x_1, x_2, x_3)],
                [y2 - function2(freeParameters[:, -1], x_1, x_2, x_3)],
            ])
            if residuals is None:
                residuals = newResiduals
            else:
                residuals = np.vstack((residuals, newResiduals))

            newAMatrix = matrix_row(freeParameters[:, -1], x_1, x_2, x_3)
            if AMatrix is None:
                AMatrix = newAMatrix
            else:
                AMatrix = np.vstack((AMatrix, newAMatrix))

        # I know the names don't mean anything, see Wolfram link
        aMatrix = AMatrix.T.dot(AMatrix)
        bMatrix = AMatrix.T.dot(residuals)
        deltaFreeParameters = np.linalg.solve(aMatrix, bMatrix)
        freeParameters = np.hstack((freeParameters,
                                    freeParameters[:, -1].reshape(6, 1) + deltaFreeParameters))

    if plotValues:
        import matplotlib.pyplot as plt
        titles = ["phi", "omega", "kappa", "s_14", "s_24", "s_34"]
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(freeParameters[i, :], "o-")
            plt.title(titles[i])
        plt.xlabel("Iterations")
        plt.show()

    # Return the best guess (most settled) values for the parameters
    return freeParameters[:, -1]


def linearLeastSquares(vertices, exteriorPts):
    # The X matrix takes certain values in certain places, see sharelatex for
    #   the details
    lenVertices = len(vertices)
    X = np.zeros((lenVertices, 12))
    for i in xrange(lenVertices):
        X[i, 0] = -f / vertices[i][0] * exteriorPts[i][0]
        X[i, 1] = -f / vertices[i][0] * exteriorPts[i][1]
        X[i, 2] = -f / vertices[i][0] * exteriorPts[i][2]       # ZERO
        X[i, 3] = -f / vertices[i][0]
        X[i, 4] = f * exteriorPts[i][0]
        X[i, 5] = f * exteriorPts[i][1]
        X[i, 6] = f * exteriorPts[i][2]
        X[i, 7] = f
        X[i, 8] = (1 - vertices[i][1]) * exteriorPts[i][0]
        X[i, 9] = (1 - vertices[i][1]) * exteriorPts[i][1]
        X[i, 10] = (1 - vertices[i][1]) * exteriorPts[i][2]     # ZERO
        X[i, 11] = (1 - vertices[i][1])
    # The Y matrix is all zeros, see sharelatex for why
    Y = np.zeros((lenVertices, 1))
    B = np.linalg.lstsq(X, Y)
    # Why is the solution always zeros?
    # When xPart is calculated X.T.dot(X) is a "singular matrix" and can't be
    #   inverted. Why?
    # The big damn problem is that X.T.dot(X) cannot be inverted if its cols
    #   are linearly Dependent, and that will happen in this case if X has cols
    #   that are linearly Dependent. Try it out in wolfram to see. Basically
    #   one X column is swept down X.T and then the negative of that is, and so
    #   the two resulting columns are just a negative off.
    def slicer(idx):
        return np.array([False if i in idx else True for i in range(12)])
    print  X[:, slicer([0, 1, 4, 5, 8, 9])].shape
    print  np.linalg.matrix_rank(X[:, slicer([0, 1, 4, 5, 8, 9])])
    xPart = np.linalg.inv(X.T.dot(X)).dot(X.T)
    bByHand = xPart.dot(Y)
    # As of 07/09/2017 when this was put here the linear least squares method
    # was totally fizzled and I turned to the non-linear method
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs exterior (hand<>eye) camera calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--folder-name",
                        help="Check this folder for calibration images",
                        default=None)
    parser.add_argument("-p", "--run-pentagon-finding",
                        help="Runs the pentagon finding code",
                        action="store_true")
    parser.add_argument("-t", "--plot-results",
                        help="Plots the final pentagon w/ lines/vertices",
                        action="store_true")
    args = parser.parse_args()

    cC = cameraCalibration(args)
