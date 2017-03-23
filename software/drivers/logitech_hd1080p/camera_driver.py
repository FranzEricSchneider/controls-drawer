import argparse
import cv2
import lcm
import numpy as np
import select
import time

from utils import lcm_msgs


ACCEPTABLE_ASPECTS = [(1280, 720), (864, 480), (640, 480), (640, 360), (176, 144)]


class CameraDriver():
    def __init__(self, args):
        if (args.width, args.height) not in ACCEPTABLE_ASPECTS:
            raise ValueError("Only width/height values allowed: {}".format(ACCEPTABLE_ASPECTS))

        self.width = args.width
        self.height = args.height
        self.fps = args.fps
        self.brightness = args.brightness
        self.contrast = args.contrast
        self.outputChannel = args.output_channel

        # TODO: Extract the calibration parameters here and undistort image

        self.videoCapture = setup(args)
        self.frame = None
        self.lcmobj = lcm.LCM()
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
        frame = self.frame
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


def setup(args):
    videoCapture = cv2.VideoCapture(args.video_index)
    if not videoCapture.isOpened():
        raise Exception("The camera is not plugged in")
    # First, attempt to set the settings
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    videoCapture.set(cv2.CAP_PROP_FPS, args.fps)
    videoCapture.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)
    videoCapture.set(cv2.CAP_PROP_CONTRAST, args.contrast)
    # Second, print what was actually set
    print("width: {}".format(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("height: {}".format(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS: {}".format(videoCapture.get(cv2.CAP_PROP_FPS)))
    print("Brightness: {}".format(videoCapture.get(cv2.CAP_PROP_BRIGHTNESS)))
    print("Contrast: {}".format(videoCapture.get(cv2.CAP_PROP_CONTRAST)))
    # Return the object
    return videoCapture


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts the camera driver, emits images when requested",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--brightness",
                        help="Contrast setting of image",
                        type=float,
                        default=0.5)
    parser.add_argument("-c", "--contrast",
                        help="Brightness setting of image",
                        type=float,
                        default=0.125)
    parser.add_argument("-f", "--fps",
                        help="Frames per second of video (actually comes in steps of ~5)",
                        type=int,
                        default=30)
    parser.add_argument("-o", "--output-channel",
                        help="LCM channel that all raw images will go out on",
                        default="IMAGE_RAW")
    parser.add_argument("-r", "--request-channel",
                        help="LCM channel that requests will come on",
                        default="REQUEST_IMAGE")
    parser.add_argument("-t", "--height",
                        help="Height of captured image in pixels",
                        type=int,
                        default=480)
    parser.add_argument("-w", "--width",
                        help="Width of captured image in pixels",
                        type=int,
                        default=864)
    parser.add_argument("-v", "--video-index",
                        help="Index of video device (tried making udev symlink but failed)",
                        type=int,
                        default=1)
    args = parser.parse_args()
    
    CD = CameraDriver(args)
    CD.run()
