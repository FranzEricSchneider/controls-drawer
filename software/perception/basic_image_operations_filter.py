import argparse
import cv2
import lcm
import numpy as np
import select
import time

from utils import lcm_msgs


class BasicOpsFilter():
    def __init__(self, args):
        self.lcmobj = lcm.LCM()
        self.requestSub = self.lcmobj.subscribe(args.input_channel,
                                                self.handleRequest)

    def run(self, loopTimeout=0.05):
        while(True):
            # This is blocking, which is fine because all this does is wait
            #   until an image comes in and then handle it, wait for the next
            self.lcmobj.handle()

    def handleRequest(self, channel, data):
        # Create the basic msg variables
        inMsg = lcm_msgs.auto_decode(channel, data)
        outMsg = lcm_msgs.auto_instantiate(inMsg.request.dest_channel)
        print("Received image on {}, re-publishing on {}"
              "".format(channel, inMsg.request.dest_channel))
        # First off, copy the old message into the new
        outMsg = inMsg
        if len(inMsg.request.arg_names) > 0:
            if "crop" in inMsg.request.arg_names:
                # Get the data as a frame
                frame = lcm_msgs.image_t_to_nparray(outMsg)
                # Get the crop values. Format is "mx,my,Mx,My"
                corners = inMsg.request.arg_values[
                    inMsg.request.arg_names.index("crop")
                ].split(",")
                minX, minY, maxX, maxY = (int(field) for field in corners)
                assert(np.all([m >= 0 for m in [minX, minY, maxX, maxY]]))
                assert(np.all([m <= (inMsg.width - 1) for m in [minX, maxX]]))
                assert(np.all([m <= (inMsg.height - 1) for m in [minY, maxY]]))
                # Crop down the array
                frame = frame[minY:maxY + 1, minX:maxX + 1]
                # Reset the image metadata
                outMsg.width = maxX - minX + 1
                outMsg.height = maxY - minY + 1
                assert(outMsg.width <= inMsg.width)
                assert(outMsg.height <= inMsg.height)
                # Reset the image data
                outMsg.data = lcm_msgs.nparray_to_image_t_data(frame)
                outMsg.num_data = len(outMsg.data)
            if "threshold" in inMsg.request.arg_names:
                # Get the data as a frame
                frame = lcm_msgs.image_t_to_nparray(outMsg)
                # Get the threshold values. The formet is EITHER "int" (0-255)
                # or "otsu". Otsu thresholding is an opencv special case where
                # an algorithm tries to split the image into background and
                # foreground. It seems to work well on the paper
                # https://en.wikipedia.org/wiki/Otsu%27s_method
                threshold = inMsg.request.arg_values[
                    inMsg.request.arg_names.index("threshold")
                ]
                if threshold.lower() == "otsu":
                    threshold, frame = cv2.threshold(frame,
                                                     thresh=0,
                                                     maxval=255,
                                                     type=cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                    outMsg.data = lcm_msgs.nparray_to_image_t_data(frame)
                    outMsg.num_data = len(outMsg.data)
                else:
                    threshold = int(threshold)
                    assert threshold >= 0
                    assert threshold <= 255
                    _, frame = cv2.threshold(frame,
                                                     thresh=threshold,
                                                     maxval=255,
                                                     type=cv2.THRESH_BINARY)
                    outMsg.data = lcm_msgs.nparray_to_image_t_data(frame)
                    outMsg.num_data = len(outMsg.data)

        # Publish the out image. If no operations were requested, could be same as input
        self.lcmobj.publish(inMsg.request.dest_channel, outMsg.encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Listens to raw image LCM and does basic requested operations",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-channel",
                        help="LCM channel that all raw images come in on",
                        default="IMAGE_RAW")
    args = parser.parse_args()
    
    BOF = BasicOpsFilter(args)
    BOF.run()
