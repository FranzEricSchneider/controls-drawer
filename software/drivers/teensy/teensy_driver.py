import argparse
import lcm
import numpy as np
import serial
import time

from utils import lcm_msgs

# Maps the index of the printed data to variable name. The variable names are
# left out of the printed line to save superfluous printing
DATA_MAPPING = {
    0: "low_limit_switch_x",
    1: "high_limit_switch_x",
    2: "low_limit_switch_y",
    3: "high_limit_switch_y",
    4: "encoder_x",
    5: "encoder_y",
}


def teensyDriver(args):
    lcmobj = lcm.LCM()
    port = serial.Serial(args.port, args.baud_rate)

    lastReadTime = time.time()
    while abs(time.time() - lastReadTime) < args.timeout:
        # Read the line and mark the time
        line = port.readline().strip()
        lastReadTime = time.time()

        # Parse the line into data chunks. They should all be ints and floats.
        # Then check that we have the right amount of data
        data = [eval(number) for number in line.split(",")]
        if len(data) != len(DATA_MAPPING):
            print("Found line with the wrong amount of data! {}".format(line))

        # Pack the raw information into the message
        msg = lcm_msgs.auto_instantiate(args.channel)
        for idx, variable in DATA_MAPPING.items():
            setattr(msg, variable, data[idx])

        # Calculate the remaining information in the message
        msg.tool_frame = np.eye(4)
        msg.camera_frame = np.eye(4)

        # Publish the message
        lcmobj.publish(args.channel, msg.encode())

    raise RuntimeError("The Teensy has not reported in {} seconds, time to"
                       " die!".format(args.timeout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs the computer-side Teensy driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--baud-rate",
                        help="Serial port baud rate, should match teensy code",
                        type=int,
                        default=115200)
    parser.add_argument("-c", "--channel",
                        help="The channel on which to publish the table data",
                        default="TABLE_STATE")
    parser.add_argument("-p", "--port",
                        help="Name of the port to connect to",
                        default="/dev/teensy")
    parser.add_argument("-t", "--timeout",
                        help="Number of seconds without data before raising",
                        type=float,
                        default=3.0)
    args = parser.parse_args()
    teensyDriver(args)
