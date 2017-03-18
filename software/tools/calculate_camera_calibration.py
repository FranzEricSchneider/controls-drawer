###############################################################################
# Runs the functions that make camera calibration happen
###############################################################################

import argparse
from glob import glob
from os.path import basename
from os.path import isfile
from os.path import join

from util.image_tools import calibrationLines
from utils.navigation import resultsDirectory
from utils.navigation import validDirectory
from utils.ui import AskAboutLinesToMerge
from utils.ui import AskAboutLinesToRemove
from utils.ui import AskAboutPentagonLines


def cameraCalibration(args):
    # Finds the folder with the calibration images
    if args.folder_name is None:
        # If you aren't given a folder, check all the calibration image folders
        # in the results directory
        resultsDirectory = resultsDirectory()
        imageDirectory = join(resultsDirectory, "calibration_images")
        searchString = join(imageDirectory, "frames*")
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
        folder = validDirectory(args.folder_name)
    else:
        # Check if the path is in results/calibration_images/
        resultsDirectory = resultsDirectory()
        imageDirectory = join(resultsDirectory, "calibration_images")
        folder = validDirectory(join(imageDirectory, args.folder_name))

    # Find the images that we want to run this on
    imagePaths = glob(join(folder, "*.png"))
    if len(imagePaths) < 1:
        raise RuntimeError("No images in folder {}".format(folder))

    for imagePath in imagePaths:
        # Runs the pentagon finding code IF there is no metadata file OR if the
        # args decree it
        imageName = basename(imagePath)
        metadata = join(folder, "pentagon_" + imageName)
        if not isfile(metadata) or args.run_pentagon_finding:
            # Get the image and all available lines
            image, finiteLines = calibrationResults(imagePath)
            # Get the starting indices for the lines
            indices = range(len(finiteLines))

            # Remove lines the user doesn't like
            lineRemover = AskAboutLinesToRemove(image, finiteLines, indices)
            lineRemover.processImage()
            indices = lineRemover.getIndices()
            finiteLines = lineRemover.getLines()

            # Merge lines the user wants to merge
            lineMerger = AskAboutLinesToMerge(image, finiteLines, indices)
            lineMerger.processImage()
            indices = lineMerger.getIndices()
            finiteLines = lineMerger.getLines()

            # Have the user identify which pentagon sides go with which lines
            pentagon = AskAboutPentagonLines(image, finiteLines, indices)
            pentagon.processImage()
            indicesBySide = pentagon.getIndicesBySide()

            # TODO: Calculate the pentagon points in pixel coordinates and write
            #       to the metadata file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs exterior (hand<>eye) camera calibration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--folder-name",
                        help="Check this folder for calibration images",
                        default=None)
    parser.add_argument("-p", "--run-pentagon-finding",
                        help="Runs the pentagon finding code",
                        action="store_true")
    args = parser.parse_args()

    cC = cameraCalibration(args)
