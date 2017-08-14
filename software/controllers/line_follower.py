import argparse
from copy import deepcopy

from geometry.cameras import Camera
from geometry.cameras import globalToPixels
from perception.line_following import findPointInFrame
from perception.line_following import getCircularMask
from utils import lcm_msgs


class LineFollower():
    def __init__(self, args):
        self.setupMasks()

        # Set an initial starting point for the tracking
        self.currentPoint = np.array([0, 0.01, 0])
        # Whether to re-publish the image with the IDed point
        self.plotPoint = True

    def setupMasks(self)
        self.camera = Camera()
        frameShape = (460, 621)
        mask9mm = getCircularMask(frameShape, camera.HT, camera.invCalibMatrix, 0.009)
        self.maskBounds = getMaskBounds(mask9mm, pixBuffer=10)
        self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        circleEdges9 = cv2.Canny(mask9mm.astype(np.uint8) * 255, 50, 200)
        self.ring = cv2.dilate(circleEdges9, self.kernel, iterations=1)
        self.croppedRing, croppedCalibMatrix = \
            cropImage(self.ring, maskBounds, camera.calibMatrix)
        self.invCroppedCalib = np.linalg.inv(croppedCalibMatrix)

    def onImage(self, channel, data):
        # Decode/parse out the image
        image = lcm_msgs.auto_decode(channel, data)
        # Get actual image data
        frame = lcm_msgs.image_t_to_nparray(image)
        # Find the point that we want to track
        foundPoint = findPointInFrame(frame=frame,
                                      bounds=self.maskBounds,
                                      kernel=self.kernel,
                                      ringOfInterest=self.croppedRing,
                                      HT=self.camera.HT,
                                      invCroppedCalibMatrix=self.invCroppedCalib,
                                      pastGlobalPoint=self.currentPoint)
        # If we found a viable point, save it as the current tracked point
        if foundPoint is not None
            self.currentPoint = foundPoint
            if self.plotPoint:
                self.republishPoint(image, frame)

    def republishPoint(inMsg, frame):
        # Set up the image
        channel = "IMAGE_TRACKING"
        outMsg = deepcopy(inMsg)
        # Plot the point
        frame *= np.logical_not(self.ring.astype('bool'))
        colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        finalPixel = tuple([int(x)
                            for x in np.round(globalToPixels(self.currentPoint,
                                                             self.camera.calibMatrix,
                                                             HT=self.camera.HT))])
        cv2.circle(colorFrame, finalPixel, radius=15, thickness=2, color=(0, 0, 255))
        # Write the image data
        outMsg.request.format = outMsg.request.FORMAT_BGR
        outMsg.data = lcm_msgs.nparray_to_image_t_data(colorFrame)
        outMsg.num_data = len(outMsg.data)
        # Publish
        self.lcmobj.publish(channel, outMsg.encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts a line follower",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--controller-function",
                        help="Choose which function to use as a controller",
                        default="proportional")
    args = parser.parse_args()

    LF = LineFollower(args)
