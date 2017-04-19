import argparse
import cv2
import glob
import lcm
import numpy as np
import pickle
import time

from utils import lcm_msgs


class CameraDriver():
    def __init__(self, args):

        self.outputChannel = args.output_channel
        self.videoCapture = self.setup(args)
        self.frame = None
        self.lcmobj = lcm.LCM()

        # Get all calibration files and use the latest to get the data
        # returned by internal calibration
        # NOTE: See drivers/general_camera_tests/test_calibration.py for
        #       explanations about these variables
        calibrationFiles = glob.glob("calibration_results*.pickle")
        calibrationResults = pickle.load(open(calibrationFiles[-1], "rb"))
        self.retval = calibrationResults["retval"]
        self.matrix = calibrationResults["matrix"]
        self.distCoeffs = calibrationResults["distCoeffs"]
        self.rvecs = calibrationResults["rvecs"]
        self.tvecs = calibrationResults["tvecs"]
        self.newCameraMatrix, self.roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.matrix,
            distCoeffs=self.distCoeffs,
            imageSize=(int(self.width), int(self.height)),
            alpha=1,
            newImgSize=(int(self.width), int(self.height))
        )
        # Overwrite the camera's width and height with the ROI width/height,
        # b/c that will be the final output
        self.width = self.roi[2]
        self.height = self.roi[3]

        self.requestSub = self.lcmobj.subscribe(args.request_channel,
                                                self.handleRequest)

    def run(self, loopTimeout=0.01):
        try:
            while(True):
                # Wait for timeout to handle lcmobj or not
                lcm_msgs.lcmobj_handle_msg(self.lcmobj, loopTimeout)

                # Capture frame-by-frame. This is blocking
                returnValue, self.frame = self.videoCapture.read()
        finally:
            # When everything done, release the capture
            self.videoCapture.release()

    def handleRequest(self, channel, data):
        print("Got image request!")

        # Grab the most recent image
        rawFrame = self.frame
        # Undistort the image re: internal calibration
        undistortedFrame = cv2.undistort(rawFrame, self.matrix, self.distCoeffs,
                                         None, self.newCameraMatrix)
        # Crop the undistorted image
        x, y, w, h = self.roi
        frame = undistortedFrame[y:y + h, x:x + w]
        # Create the basic msg variables
        inMsg = lcm_msgs.auto_decode(channel, data)
        outMsg = lcm_msgs.auto_instantiate(self.outputChannel)
        # Start filling in the data that isn't related to the image
        outMsg.utime = lcm_msgs.utime_now()
        outMsg.action_id = inMsg.action_id
        outMsg.request = inMsg
        # If necessary, convert to grayscale
        if inMsg.format == inMsg.FORMAT_GRAY:
            # Make the image grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Keep the array order the same as the color image
            frame = np.expand_dims(frame, axis=2)
        # Write the fixed image parameters
        outMsg.width = self.width
        outMsg.height = self.height
        # strides is the number of bytes until the next element of that axis.
        #   In one example, I got frame.strides = (2592, 3, 1). This means that
        #   it is 2592 bytes to traverse to the next row, 3 bytes to traverse to
        #   the next column, and 1 bytes to traverse to the next element in the
        #   pixel. This makes sense b/c their are 3 numbers per column element
        #   in an RGB image. row_stride is strides[0]
        outMsg.row_stride = frame.strides[0]
        outMsg.FPS = self.fps
        outMsg.brightness = self.brightness
        outMsg.contrast = self.contrast
        # Write the image data
        outMsg.data = lcm_msgs.nparray_to_image_t_data(frame)
        outMsg.num_data = len(outMsg.data)
        # Publish the raw image!
        print("Publishing on {}".format(self.outputChannel))
        self.lcmobj.publish(self.outputChannel, outMsg.encode())

    def setup(self, args):
        videoCapture = cv2.VideoCapture(args.video_index)
        if not videoCapture.isOpened():
            raise Exception("The camera is not plugged in")
        self.width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = videoCapture.get(cv2.CAP_PROP_FPS)
        self.brightness = videoCapture.get(cv2.CAP_PROP_BRIGHTNESS)
        self.contrast = videoCapture.get(cv2.CAP_PROP_CONTRAST)
        # Print the camera settings
        print("width: {}".format(self.width))
        print("height: {}".format(self.height))
        print("FPS: {}".format(self.fps))
        print("Brightness: {}".format(self.brightness))
        print("Contrast: {}".format(self.contrast))
        # Return the object
        return videoCapture


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts the camera driver, emits images when requested",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output-channel",
                        help="LCM channel that all raw images will go out on",
                        default="IMAGE_RAW")
    parser.add_argument("-r", "--request-channel",
                        help="LCM channel that requests will come on",
                        default="REQUEST_IMAGE")
    parser.add_argument("-v", "--video-index",
                        help="Index of video device (tried making udev symlink but failed)",
                        type=int,
                        default=1)
    args = parser.parse_args()
    
    CD = CameraDriver(args)
    CD.run()
