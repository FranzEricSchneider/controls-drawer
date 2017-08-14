import argparse
import cv2
import lcm
import numpy as np
import time

from utils import lcm_msgs


def main(args):
    # Set up the basic LCM tools
    lcmobj = lcm.LCM()

    # If we want to display incoming images, we need to catch them
    if args.visualize:
        lcmobj.subscribe(args.subscribe_channel, displayImage)

    # Build the message
    channel = args.request_channel
    msg = lcm_msgs.auto_instantiate(channel)
    msg.dest_channel = args.destination_channel
    if args.color:
        msg.format = msg.FORMAT_BGR
    else:
        msg.format = msg.FORMAT_GRAY
    if args.crop_arguments != "":
        msg.arg_names.append("crop")
        msg.arg_values.append(args.crop_arguments)
    if args.threshold_arguments != "":
        msg.arg_names.append("threshold")
        msg.arg_values.append(args.threshold_arguments)
    msg.n_arguments = len(msg.arg_names)

    # Publish the messages as commanded
    if args.stream:
        sleepTime = 1.0 / args.stream_rate
        while True:
            msg.utime = lcm_msgs.utime_now()
            lcmobj.publish(channel, msg.encode())
            time.sleep(sleepTime)
    else:
        lcmobj.publish(channel, msg.encode())
        if args.visualize:
            lcmobj.handle()


def displayImage(channel, data):
    msg = lcm_msgs.auto_decode(channel, data)
    frame = lcm_msgs.image_t_to_nparray(msg)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool to send image requests",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--crop-arguments",
                        help="If crop coordinates given, pass to basic_image_ops_filter (iMin,iMax,jMin,jMax)",
                        default="")
    parser.add_argument("-d", "--destination-channel",
                        help="End-goal LCM channel for images",
                        default="IMAGE_NOOP")
    parser.add_argument("-l", "--color",
                        help="Requests a color image",
                        action="store_true")
    parser.add_argument("-o", "--threshold-arguments",
                        help="If threshold given, pass to basic_image_ops_filter (0-255 or otsu)",
                        default="")
    parser.add_argument("-r", "--request-channel",
                        help="LCM channel that requests will be sent on",
                        default="REQUEST_IMAGE")
    parser.add_argument("-s", "--subscribe-channel",
                        help="LCM channel to listen for returns on",
                        default="IMAGE_RAW")
    parser.add_argument("-t", "--stream",
                        help="Stream image requests at a certain frequency",
                        action="store_true")
    parser.add_argument("-T", "--stream-rate",
                        help="Rate at which to stream requests, if stream is on",
                        type=float,
                        default=10.0)
    parser.add_argument("-v", "--visualize",
                        help="Display images sent on subscribe-channel",
                        action="store_true")
    args = parser.parse_args()

    if args.stream and args.visualize:
        raise ValueError("Visualize and stream were set to happen at the same"
                         " time, but they are incompatible")

    main(args)
