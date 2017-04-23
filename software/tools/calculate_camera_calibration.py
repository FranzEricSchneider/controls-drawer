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
from utils import cv_tools
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
    # The X matrix takes certain values in certain places, see sharelatex for
    #   the details
    lenVertices = len(vertices)
    X = np.zeros((lenVertices, 12))
    for i in xrange(lenVertices):
        X[i, 0] = -f * exteriorPts[i][0]
        X[i, 1] = -f * exteriorPts[i][1]
        X[i, 2] = -f * exteriorPts[i][2]
        X[i, 3] = -f
        X[i, 4] = f * exteriorPts[i][0]
        X[i, 5] = f * exteriorPts[i][1]
        X[i, 6] = f * exteriorPts[i][1]
        X[i, 7] = f
        X[i, 8] = (vertices[i][0] - vertices[i][1]) * exteriorPts[i][0]
        X[i, 9] = (vertices[i][0] - vertices[i][1]) * exteriorPts[i][1]
        X[i, 10] = (vertices[i][0] - vertices[i][1]) * exteriorPts[i][2]
        X[i, 11] = (vertices[i][0] - vertices[i][1])
    # The Y matrix is all zeros, see sharelatex for why
    Y = np.zeros((lenVertices, 1))
    B = np.linalg.lstsq(X, Y)
    # Why is the solution always zeros?
    # When xPart is calculated X.T.dot(X) is a "singular matrix" and can't be
    #   inverted. Why?
    import ipdb; ipdb.set_trace()
    xPart = np.linalg.inv(X.T.dot(X)).dot(X.T)
    bByHand = xPart.dot(Y)


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
        pointsFromCamera = pentagonVectors * sideLength - np.array([x, y, 0])
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
