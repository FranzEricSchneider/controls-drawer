###############################################################################
# Helps to find folders and files in the repo
###############################################################################

import os
import glob


def baseDir():
    base = os.environ['CNC_DRAWER_BASE']
    # Throws an error if the directory is invalid
    return validDir(base)


def resultsDir():
    results = os.environ['CNC_DRAWER_RESULTS']
    # Throws an error if the directory is invalid
    return validDir(results)


def softwareDir():
    software = os.environ['CNC_DRAWER_SOFTWARE']
    # Throws an error if the directory is invalid
    return validDir(software)


def validDir(directory):
    if not os.path.isdir(directory):
        raise OSError("Hey! {} isn't a directory!".format(directory))
    if not os.access(directory, os.R_OK):
        raise OSError("Hey! Can't read {} directory".format(directory))
    return directory


def getLatestIntrinsicCalibration():
    searchString = os.path.join(
        softwareDir(),
        "drivers/crenova_iscope_endoscope_2Mpix/intrinsic*.pickle"
    )
    return glob.glob(searchString)[-1]
