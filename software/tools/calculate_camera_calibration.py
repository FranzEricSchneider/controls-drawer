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
    focalX = calibrationResults['matrix'][0, 0]
    focalY = calibrationResults['matrix'][1, 1]
    # Take the average of x and y, they are already almost equal
    focalLength = (focalX + focalY) / 2.0
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
    freeParameters = nonLinearLeastSquares(focalLength, imFrameVertices, exteriorPts,
                                           args.plot_parameters)
    HT = HT_from_parameters(freeParameters)
    geometry_tools.checkOrthonormal(HT)

    if args.plot_final_results:
        for imagePath in imagePaths:
        # for imagePath in [imagePaths[0]]:
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
                cameraFramePixels = globalToPixels(globalFramePt, HT, calibrationResults['matrix'])
                pointCenter = tuple([int(x) for x in cameraFramePixels])
                cv2.circle(image, pointCenter, radius=4, thickness=t,
                           color=(0, 0, 255))

                cv2.line(image, vertexCenter, pointCenter, color=(0, 255, 0))
                # import ipdb; ipdb.set_trace()

            cv_tools.showImage(metadata, image)

    if args.final_3d_plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        exteriorPtsXYZ = np.array(exteriorPts)
        exteriorPtsXYZ = np.hstack((exteriorPtsXYZ, np.ones((exteriorPtsXYZ.shape[0], 1))))
        cameraFramePtsXYZ = HT.dot(exteriorPtsXYZ.T).T
        import ipdb; ipdb.set_trace()
        pass

        for counter in np.arange(5, len(exteriorPts) + 5, 5):
            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')
            # axes.scatter(xs=exteriorPtsXYZ[:counter, 0],
            #              ys=exteriorPtsXYZ[:counter, 1],
            #              zs=exteriorPtsXYZ[:counter, 2])
            axes.scatter(xs=cameraFramePtsXYZ[:counter, 0],
                         ys=cameraFramePtsXYZ[:counter, 1],
                         zs=cameraFramePtsXYZ[:counter, 2])
            # axes.set_xlim(-0.04, 0.04)
            # axes.set_ylim(-0.04, 0.04)
            # axes.set_zlim(-0.04, 0.04)
            plt.show()
            import ipdb; ipdb.set_trace()

    if args.plot_axes:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        origin = np.array([0, 0, 0, 1])
        camOrigin = HT[:, 3]
        vectors = np.array([[0.1, 0, 0, 1], [0, 0.1, 0, 1], [0, 0, 0.1, 1]])
        camVectors = HT.dot(vectors.T).T

        axes.plot(xs=[origin[0], vectors[0, 0]],
                  ys=[origin[1], vectors[0, 1]],
                  zs=[origin[2], vectors[0, 2]], color=(1.0, 0, 0))
        axes.plot(xs=[origin[0], vectors[1, 0]],
                  ys=[origin[1], vectors[1, 1]],
                  zs=[origin[2], vectors[1, 2]], color=(0, 1.0, 0))
        axes.plot(xs=[origin[0], vectors[2, 0]],
                  ys=[origin[1], vectors[2, 1]],
                  zs=[origin[2], vectors[2, 2]], color=(0, 0, 1.0))

        axes.plot(xs=[camOrigin[0], camVectors[0, 0]],
                  ys=[camOrigin[1], camVectors[0, 1]],
                  zs=[camOrigin[2], camVectors[0, 2]], color=(1.0, 0, 0))
        axes.plot(xs=[camOrigin[0], camVectors[1, 0]],
                  ys=[camOrigin[1], camVectors[1, 1]],
                  zs=[camOrigin[2], camVectors[1, 2]], color=(0, 1.0, 0))
        axes.plot(xs=[camOrigin[0], camVectors[2, 0]],
                  ys=[camOrigin[1], camVectors[2, 1]],
                  zs=[camOrigin[2], camVectors[2, 2]], color=(0, 0, 1.0))

        # axes.set_xlim(-0.1, 0.35)
        # axes.set_ylim(-0.1, 0.35)
        # axes.set_zlim(-0.1, 0.35)
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

        imFrameVertices.extend([pixelsToImFrame(pixel, calibMatrix) for pixel in data["pixelVertices"]])

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
        pentagonAddition = np.cumsum(pentagonVectors * sideLength, axis=0)
        # The subtraction is because the (x,y) points show how much the toolframe
        # moved before the picture. Ifframe moved (for example) -0.01 meters,
        # then the points are now 0.01 in the positive direction from the toolframe
        pointsFromCamera = pentagonAddition - np.array([x, y, 0.0])
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


def nonLinearLeastSquares(f, vertices, exteriorPts, plotValues=False):
    # Method and many of the more meaningless names taken from here:
    # http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html

    # Choose initial values for the free parameters
    phi = np.pi
    omega = 0.0
    kappa = -1.0
    s_14 = 0.01
    s_24 = 0.02
    s_34 = 0.07
    # Track the parameters over time in a matrix, use the latest values to
    # calculate each consecutive step
    freeParameters = np.array([phi, omega, kappa, s_14, s_24, s_34]).reshape(6, 1)

    # TODO: Make this for loop a combination of delta resolution and maximum
    #       iterations
    for i in range(50):
        # Loop through every measurement point
        residuals = None
        AMatrix = None
        for i in xrange(len(vertices)):
            x_1 = exteriorPts[i][0]
            x_2 = exteriorPts[i][1]
            x_3 = exteriorPts[i][2]

            # y1 and y2 are the "measured output" variables, the (x,y) values
            # in the image frame
            y1 = vertices[i][0]
            y2 = vertices[i][1]

            # Calculate current residuals
            newResiduals = np.array([
                [y1 - function1(freeParameters[:, -1], x_1, x_2, x_3, f)],
                [y2 - function2(freeParameters[:, -1], x_1, x_2, x_3, f)],
            ])
            if residuals is None:
                residuals = newResiduals
            else:
                residuals = np.vstack((residuals, newResiduals))

            newAMatrix = matrix_row(freeParameters[:, -1], x_1, x_2, x_3, f)
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


def pixelsToImFrame(pixelPoint, calibMatrix):
    # TODO: Assumptions about point
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if len(pixelPoint) == 2:
        pixelPoint = np.hstack((pixelPoint, 1))
    return np.linalg.inv(calibMatrix).dot(pixelPoint)


def globalToPixels(point, HT, calibMatrix):
    # TODO: Assumptions about point
    # See details here:
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    if len(point) == 3:
        point = np.hstack((point, 1))
    cameraFramePt = HT.dot(point)
    unscaledPixels = calibMatrix.dot(cameraFramePt[0:3])
    unitPixels = unscaledPixels / unscaledPixels[2]
    return unitPixels[0:2]


# def linearLeastSquares(vertices, exteriorPts):
#     # The X matrix takes certain values in certain places, see sharelatex for
#     #   the details
#     lenVertices = len(vertices)
#     X = np.zeros((lenVertices, 12))
#     for i in xrange(lenVertices):
#         X[i, 0] = -f / vertices[i][0] * exteriorPts[i][0]
#         X[i, 1] = -f / vertices[i][0] * exteriorPts[i][1]
#         X[i, 2] = -f / vertices[i][0] * exteriorPts[i][2]       # ZERO
#         X[i, 3] = -f / vertices[i][0]
#         X[i, 4] = f * exteriorPts[i][0]
#         X[i, 5] = f * exteriorPts[i][1]
#         X[i, 6] = f * exteriorPts[i][2]
#         X[i, 7] = f
#         X[i, 8] = (1 - vertices[i][1]) * exteriorPts[i][0]
#         X[i, 9] = (1 - vertices[i][1]) * exteriorPts[i][1]
#         X[i, 10] = (1 - vertices[i][1]) * exteriorPts[i][2]     # ZERO
#         X[i, 11] = (1 - vertices[i][1])
#     # The Y matrix is all zeros, see sharelatex for why
#     Y = np.zeros((lenVertices, 1))
#     B = np.linalg.lstsq(X, Y)
#     # Why is the solution always zeros?
#     # When xPart is calculated X.T.dot(X) is a "singular matrix" and can't be
#     #   inverted. Why?
#     # The big damn problem is that X.T.dot(X) cannot be inverted if its cols
#     #   are linearly Dependent, and that will happen in this case if X has cols
#     #   that are linearly Dependent. Try it out in wolfram to see. Basically
#     #   one X column is swept down X.T and then the negative of that is, and so
#     #   the two resulting columns are just a negative off.
#     def slicer(idx):
#         return np.array([False if i in idx else True for i in range(12)])
#     print  X[:, slicer([0, 1, 4, 5, 8, 9])].shape
#     print  np.linalg.matrix_rank(X[:, slicer([0, 1, 4, 5, 8, 9])])
#     xPart = np.linalg.inv(X.T.dot(X)).dot(X.T)
#     bByHand = xPart.dot(Y)
#     # As of 07/09/2017 when this was put here the linear least squares method
#     # was totally fizzled and I turned to the non-linear method
#     return None


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
