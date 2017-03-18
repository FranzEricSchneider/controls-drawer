###############################################################################
# Helps to find folders and files in the repo
###############################################################################

import os


def baseDirectory():
    base = os.environ['CNC_DRAWER_BASE']
    # Throws an error if the directory is invalid
    return validDirectory(base)


def resultsDirectory():
    results = os.environ['CNC_DRAWER_RESULTS']
    # Throws an error if the directory is invalid
    return validDirectory(results)


def softwareDirectory():
    software = os.environ['CNC_DRAWER_SOFTWARE']
    # Throws an error if the directory is invalid
    return validDirectory(software)


def validDirectory(directory):
    if not os.path.isdir(directory):
        raise OSError("Hey! {} isn't a directory!".format(directory))
    if not os.access(directory, os.R_OK):
        raise OSError("Hey! Can't read {} directory".format(directory))
    return directory
