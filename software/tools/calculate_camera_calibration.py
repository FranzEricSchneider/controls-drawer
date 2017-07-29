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
from geometry import cameras
from perception import free_parameter_eqs
from utils import cv_tools
from utils import geometry_tools
from utils import navigation
from utils.geometry_tools import plotAxes
from utils import ui


def cameraCalibration(args):
    # Finds the directory with the calibration images
    if args.directory_name is None:
        # If you aren't given a directory, check all the calibration image
        # directories in the results directory
        resultsDirectory = navigation.resultsDir()
        imageDirectory = path.join(resultsDirectory, "calibration_images")
        searchString = path.join(imageDirectory, "frames*")
        # Sort the directories - because the names should be identical except
        # for the utime (directory_name name should be 'frames_utime') this will put
        # the most recently made directories last
        possibleDirectories = sorted(glob(searchString))
        # Check that there are valid directories
        if len(possibleDirectories) < 1:
            raise RuntimeError("There are no calibration directories")
        # Set the oldest directory as the designated one
        directory = possibleDirectories[-1]
    elif args.directory_name.startswith("/home"):
        # Check and see if the path is global
        directory = navigation.validDir(args.directory_name)
    else:
        # Check if the path is in results/calibration_images/
        resultsDirectory = navigation.resultsDir()
        imageDirectory = path.join(resultsDirectory, "calibration_images")
        directory = navigation.validDir(path.join(imageDirectory,
                                               args.directory_name))

    # Find the images that we want to run this on
    imagePaths = glob(path.join(directory, "*.png"))
    if len(imagePaths) < 1:
        raise RuntimeError("No images in directory {}".format(directory))

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
            pixelVertices = [midInfiniteLines[i].intersection(midInfiniteLines[j])
                             for i, j in zip(midIndices, np.roll(midIndices, 1))]

            # Write the lines/vertices data to file
            data = {"pixelVertices": [list(vertex) for vertex in pixelVertices]}
            with open(metadata, "w") as outfile:
                json.dump(data, outfile)
            print("Wrote pixelVertices to {}".format(metadata))

            # Plot the final pentagon if desired
            if args.plot_pentagon_results:
                finalImage = image.copy()
                for line in midInfiniteLines:
                    line.onImage(finalImage, thickness=1)
                for line in midLines:
                    line.onImage(finalImage, color=(0, 255, 0), thickness=2)
                for vertex in pixelVertices:
                    center = tuple([int(x) for x in vertex])
                    cv2.circle(finalImage, center, radius=6, thickness=2,
                               color=(204, 255, 0))
                cv_tools.showImage(metadata, finalImage)

        elif args.plot_pentagon_results:
            print("Displaying pre-found pentagon {}".format(imageName))

            # Read in image and point data
            image = cv_tools.readImage(imagePath)
            with open(metadata, "r") as infile:
                data = json.load(infile)

            # Plot and show those vertices
            for vertex in data["pixelVertices"]:
                center = tuple([int(x) for x in vertex])
                cv2.circle(image, center, radius=6, thickness=2,
                           color=(204, 255, 0))
            cv_tools.showImage(metadata, image)

        else:
            pass
            # print("Found file containing point pairs for {}".format(imageName))

    # Get the interior camera calibration data to get a number for focal length
    calibrationResults = pickle.load(
        open(navigation.getLatestIntrinsicCalibration(), "rb"))
    # Time to take the pixel vertex values, real world displacement values, and
    # combine them into a transformation from tooltip to camera. Check out
    # https://www.sharelatex.com/project/586949e817ccee00403fbc56 for the math
    # behind this part
    pixelVertices, imFrameVertices, exteriorPts = \
        getCalibPoints(imagePaths, calibrationResults["matrix"])

    # fullVertices = vertices
    # fullExteriorPts = exteriorPts
    # numIterations = 10
    # for numRemove in np.arange(30, -3, -3):
    #     print("Removing {}".format(numRemove))
    #     if numRemove > 0:
    #         vertices = fullVertices[0:-numRemove]
    #         exteriorPts = fullExteriorPts[0:-numRemove]
    #     else:
    #         vertices = fullVertices
    #         exteriorPts = fullExteriorPts

    # Calculate the 6 free parameters that make up a homogeneous transform,
    # three Euler angles and three translation distances
    omega = np.pi
    phi = 0.0
    kappa = -np.pi
    s_14 = 0.0
    s_24 = 0.0
    s_34 = 0.09
    parameters = (omega, phi, kappa, s_14, s_24, s_34)
    freeParameters = free_parameter_eqs.nonLinearLeastSquares(
        imFrameVertices, exteriorPts, parameters, plotValues=args.plot_parameters
    )
    HT = free_parameter_eqs.HTFromParameters(freeParameters)
    geometry_tools.checkOrthonormal(HT)

    if args.plot_final_results:
        for imagePath in imagePaths:
            # Get the basic information necessary
            image = cv_tools.readImage(imagePath)
            pixelVertices, imFrameVertices, exteriorPts = \
                getCalibPoints([imagePath], calibrationResults['matrix'])

            # Display the original points
            for t, (vertex, point) in enumerate(zip(pixelVertices, exteriorPts)):
                vertexCenter = tuple([int(x) for x in vertex])
                cv2.circle(image, vertexCenter, radius=6, thickness=t,
                           color=(204, 255, 0))

                globalFramePt = np.hstack((point, 1.0))
                cameraFramePixels = cameras.globalToPixels(
                    globalFramePt, HT, calibrationResults['matrix']
                )
                pointCenter = tuple([int(x) for x in cameraFramePixels])
                cv2.circle(image, pointCenter, radius=4, thickness=t,
                           color=(0, 0, 255))

                # Label the exterior points with their global coordinates
                cv2.putText(img=image,
                            text="({:.4f},{:.4f})".format(point[0], point[1]),
                            org=pointCenter,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4,
                            color=(175, 0, 255))
                # Line between matching exterior points and pixel vertex
                cv2.line(image, vertexCenter, pointCenter, color=(0, 255, 0))

            cv_tools.showImage(path.basename(imagePath), image)

    if args.final_3d_plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        exteriorPtsXYZ = np.array(exteriorPts)
        exteriorPtsXYZ = np.hstack((exteriorPtsXYZ, np.ones((exteriorPtsXYZ.shape[0], 1))))
        cameraFramePtsXYZ = HT.dot(exteriorPtsXYZ.T).T

        for counter in np.arange(5, len(exteriorPts) + 5, 5):
            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')
            plotAxes(axes, np.eye(4), scalar=0.1)
            axes.scatter(xs=exteriorPtsXYZ[counter-5:counter, 0],
                         ys=exteriorPtsXYZ[counter-5:counter, 1],
                         zs=exteriorPtsXYZ[counter-5:counter, 2])
            # axes.scatter(xs=cameraFramePtsXYZ[:counter, 0],
            #              ys=cameraFramePtsXYZ[:counter, 1],
            #              zs=cameraFramePtsXYZ[:counter, 2])
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')
            # axes.set_xlim(-0.04, 0.04)
            # axes.set_ylim(-0.04, 0.04)
            # axes.set_zlim(-0.04, 0.04)
            plt.show()

    if args.plot_axes:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')

        plotAxes(axes, np.eye(4), scalar=0.1)
        plotAxes(axes, HT, scalar=0.1)

        axes.set_xlim(-0.1, 0.1)
        axes.set_ylim(-0.1, 0.1)
        axes.set_zlim(-0.1, 0.1)
        plt.show()


def getCalibPoints(imagePaths, calibMatrix):
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

    pixelVertices = []
    imFrameVertices = []
    exteriorPts = []

    for imagePath in imagePaths:
        metadata = getMetadata(imagePath)
        imageName = path.basename(imagePath)

        # Get vertices from the metadata
        with open(metadata, "r") as infile:
            data = json.load(infile)
        pixelVertices.extend(data["pixelVertices"])

        imFrameVertices.extend([cameras.pixelsToImFrame(np.array(pixel), calibMatrix)
                                for pixel in data["pixelVertices"]])

        # Parse tooltip data out of the image name
        imageNameSplit = imageName.lower().split("_")
        sideLength = None
        xTravel = None
        yTravel = None
        for part in imageNameSplit:
            if part.startswith("sl"):
                # Length of pentagon side, in meters
                sideLength = float(part.replace("sl", "")) / 1000
            if part.startswith("x"):
                # X travel from 1st point to image point, in meters
                xTravel = float(part.replace("x", "")) / 1000
            if part.startswith("y"):
                # Y travel from 1st point to image point, in meters
                yTravel = float(part.replace("y", "")) / 1000
        if sideLength is None or xTravel is None or yTravel is None:
            raise ValueError("Image {} lacks data".format(imageName))

        # Solve for the pentagon points in the exterior frame
        pentagonPoints = planar.polygonPoints(5, sideLength)
        # The subtraction is because the (x,y) points show how much the toolframe
        # moved before the picture. Ifframe moved (for example) -0.01 meters,
        # then the points are now 0.01 in the positive direction from the toolframe
        pointsFromCamera = pentagonPoints - np.array([xTravel, yTravel, 0.0])
        exteriorPts.extend(list(pointsFromCamera))

    return (pixelVertices, imFrameVertices, exteriorPts)


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
    parser.add_argument("-a", "--plot-axes",
                        help="Plots the toolframe axes against the camera axes",
                        action="store_true")
    parser.add_argument("-d", "--directory-name",
                        help="Check this directory for calibration images. If"
                             " None, run on the latest frames_{timestamp}"
                             " directory in results/calibration_images",
                        default=None)
    parser.add_argument("-f", "--run-pentagon-finding",
                        help="Runs the pentagon finding code. If there is no"
                        " metadata file for an image the pentagon finding code"
                        " is run anyway",
                        action="store_true")
    parser.add_argument("-l", "--final-3d-plot",
                        help="Displays the exterior points in 3D for debugging",
                        action="store_true")
    parser.add_argument("-p", "--plot-pentagon-results",
                        help="Plots the final pentagon w/ lines/vertices",
                        action="store_true")
    parser.add_argument("-r", "--plot-final-results",
                        help="Closes the loop and checks the initial points",
                        action="store_true")
    parser.add_argument("-t", "--plot-parameters",
                        help="Plots how the free parameters asymptote",
                        action="store_true")
    args = parser.parse_args()

    cC = cameraCalibration(args)
