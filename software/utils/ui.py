###############################################################################
# Has functions for querying the user for information (for the purposes of
# cheating, normally)
###############################################################################

from collections import defaultdict
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
        # Go in and render the lines remaining (do the lines first)
        for i in self.indices:
            self.finiteLines[i].onImage(image)
        # Label the text for each line after doing all the lines
        for i in self.indices:
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
        userList = queryList(self.actionTask,
                             allowedValues=self.indices,
                             redisplay=self.displayImage)
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
        userList = queryList(self.actionTask,
                             allowedValues=self.indices,
                             forceLen=2,
                             redisplay=self.displayImage)
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


class AskAboutPentagonLines(AskAboutImage):
    def __init__(self, image, finiteLines, indices):
        AskAboutImage.__init__(self, image, finiteLines, indices)
        self.sideNames = {0 : "(bottom side)",
                          1 : "(lower right)",
                          2 : "(upper right)",
                          3 : "(upper left)",
                          4 : "(lower left)"}
        # Stores these in the form {idx : sideNumber}
        self.sideIndices = {}

    def imageWithLines(self):
        # Make a copy for fresh display
        image = self.image.copy()
        # Go in and render the lines remaining (do the lines first)
        for i in self.indices:
            self.finiteLines[i].onImage(image)
        # Label the text for each line after doing all the lines
        for i in self.indices:
            if i in self.sideIndices.keys():
                lineText = "{}, side {}".format(i, self.sideIndices[i] + 1)
            else:
                lineText = "{}".format(i)
            self.finiteLines[i].textOnImage(image, lineText)
        return image

    def processImage(self):
        # Get the points on each side of the pentagon
        for i in xrange(5):
            print("Look for the lines that make up side {} {}"
                  "".format(i + 1, self.sideNames[i]))
            self.displayImage()
            # Get the thus unused indices
            unusedIndices = []
            for idx in self.indices:
                if idx not in self.sideIndices.keys():
                    unusedIndices.append(idx)
            # Get the lines from the user that correspond to this edge
            prompt = "Select the lines that make up side {}".format(i + 1)
            lines = queryList(prompt,
                              allowedValues=unusedIndices,
                              forceLen=2,
                              redisplay=self.displayImage)
            for lineIdx in lines:
                self.sideIndices[lineIdx] = i

    def getSidesByIndex(self):
        return self.sideIndices

    def getIndicesBySide(self):
        indicesBySide = defaultdict(list)
        # Swap the key/value order
        for k, v in self.sideIndices.items():
            indicesBySide[v].append(k)
        # Return the values as numpy arrays
        return {k : np.array(v) for k, v in indicesBySide.items()}


def queryList(prompt, allowedValues=None, allowDuplicates=False, forceLen=None,
              forceType=int, redisplay=None):
    print(prompt)
    print("(NOTE) x means delete what you have and re-start")
    print("(NOTE) r means re-see the image and restart choices")
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

        result = raw_input("Type and hit [enter] (remember x restart, q end, r"
                           " re-show image): ")
        if result.lower().startswith("x"):
            print("Alright, we had {} but we are chucking this and starting"
                  " fresh".format(valueList))
            valueList = []
        elif result.lower().startswith("r") and redisplay is not None:
            redisplay()
            return queryList(prompt=prompt,
                             allowedValues=allowedValues,
                             allowDuplicates=allowDuplicates,
                             forceLen=forceLen,
                             forceType=forceType,
                             redisplay=redisplay)
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
