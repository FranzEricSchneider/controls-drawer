################################################################################
# Has functions for querying the user for information (for the purposes of
# cheating, normally)
################################################################################

import cv2
from humanfriendly import prompts
import numpy as np


class AskAboutImage():
    def __init__(self, image, finiteLines, indices):
        self.image = image
        self.finiteLines = finiteLines
        self.indices = indices

    def displayImage(self, windowName="Look at lines of interest"):
        image = self.imageWithLines()
        # Display the image and disappear when a key is hit
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()
        cv2.imshow(windowName, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def imageWithLines(self):
        # Make a copy for fresh display
        image = self.image.copy()
        # Go in and render the lines remaining
        for i in self.indices:
            self.finiteLines[i].onImage(image)
            self.finiteLines[i].textOnImage(image, "{}".format(i))
        return image

    def processImage(self):
        print("You are about to see an image and evaluate whether you want to"
              " take action: {}".format(self.actionPrompt))
        takeAction = True
        while takeAction:
            self.displayImage()
            takeAction = prompts.prompt_for_confirmation(self.actionPrompt)
            if takeAction:
                self.processIndices()
            else:
                print("Ending {}".format(self.__class__.__name__))

    def processIndices(self):
        raise NotImplementedError

    def getIndices(self):
        return self.indices

    def getLines(self):
        return self.finiteLines


class AskAboutLinesToRemove(AskAboutImage):
    def __init__(self, image, finiteLines, indices):
        AskAboutImage.__init__(self, image, finiteLines, indices)
        self.actionPrompt = "(ACTION) Are there any lines to remove?"
        self.actionTask = "Select all line indices you would like to remove"

    def processIndices(self):
        userList = queryList(self.actionTask, allowedValues=self.indices)
        print("Removing indices {} from {}\n".format(userList, self.indices))
        # Remove indices while checking that all indices are available
        for idx in userList:
            assert idx in self.indices
            self.indices.remove(idx)


class AskAboutLinesToMerge(AskAboutImage):
    def __init__(self, image, finiteLines, indices):
        AskAboutImage.__init__(self, image, finiteLines, indices)
        self.actionPrompt = "(ACTION) Are there any lines to merge?"
        self.actionTask = "Select two lines you would like to merge"

    def processIndices(self):
        userList = queryList(self.actionTask, allowedValues=self.indices,
                             forceLen=2)
        print("Merging indices {} into one line\n".format(userList))
        # Check that we are only merging two lines
        assert len(userList) == 2
        # Remove indices while checking that all indices are available
        for idx in userList:
            assert idx in self.indices
            self.indices.remove(idx)
        # Add a new index for a new line - the combination 
        self.indices.append(len(self.finiteLines))
        self.finiteLines.append(
            self.finiteLines[userList[0]].average(
                self.finiteLines[userList[1]]
            )
        )


# class AskAboutPentagonLines(AskAboutImage):


def queryList(prompt, allowedValues=None, allowDuplicates=False, forceLen=None,
              forceType=int):
    print(prompt)
    print("(NOTE) x means delete what you have and re-start")
    if forceLen is None:
        print("(NOTE) q means done")
    else:
        print("(NOTE) You must enter a valid list of length {}".format(forceLen))

    done = False
    valueList = []
    while not done:
        print("Current list: {}".format(valueList))
        if forceLen is not None and len(valueList) == forceLen:
            print("Alright, done! Current list has {0}/{0} values"
                  "".format(len(valueList)))
            done = True
            continue

        result = raw_input("Type and hit [enter] (remember x to restart, q to"
                           " end if applicable): ")
        if result.lower().startswith("x"):
            print("Alright, we had {} but we are chucking this and starting"
                  " fresh".format(valueList))
            valueList = []
        elif forceLen is None and result.lower().startswith("q"):
            print("Alright, done! Taking the current list")
            done = True
        else:
            try:
                value = forceType(result)
                if allowedValues is not None and value not in allowedValues:
                    print("You entered {} but that was not in allowed list {}"
                          "".format(result, allowedValues))
                elif not allowDuplicates and value in valueList:
                    print("You entered {} but duplicates are not allowed and"
                          " the list is already {}".format(result, valueList))
                else:
                    valueList.append(value)
            except ValueError:
                print("You entered {} but that could not be cast into"
                      " {}".format(result, forceType))

    return valueList
