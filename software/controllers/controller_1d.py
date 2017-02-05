#!/usr/bin/env python

import argparse
import numpy as np

import lcm

from lcmtypes import lcm_velocity_t


DEBUG_HANDLE = False

CONSTANT_FN_VALUE_M = 0.05

SINE_FREQ_RPS = np.pi / 8  # /2pi = cycles per second (1/16 cycle per second)
SINE_AMPLITUDE_M = 0.01
SINE_OFFSET_M = 0.0
SINE_ANGULAR_OFFSET_R = np.pi

SAWTOOTH_SLOPE = 0.002
SAWTOOTH_Y_LIMIT_M = 0.03

SQUARE_WIDTH_S = 3
SQUARE_HEIGHT_M = 0.02

# It's wierd. The x axis is measured in time and the y axis in meters
CIRCLE_RADIUS_S = 3.0
CIRCLE_RADIUS_M = 0.005
CIRCLE_INVERT = True

P_CONSTANT = 2.5  # 1cm offset results in reaction of 0.005 m/s, 300 mm/min


class Controller1D():
    def __init__(self, referencePointFn, controllerFn, timestep):
        self.referencePointFn = referencePointFn
        self.controllerFn = controllerFn
        self.timestep = timestep

        self.lcmObj = lcm.LCM()
        self.subscription = self.lcmObj.subscribe("HEAD_POSITION",
                                                  self.positionHandler)
        self.cycleStartUtime = long(0)
        self.timeLastSent = time.time()
        self.plotX1 = []
        self.plotY1 = []
        self.plotX2 = []
        self.plotY2 = []

        try:
           while True:
                try:
                    self.lcmObj.handle()
                except:
                    # This is necessary because otherwise lcmObj goes through
                    #   the outer try except
                    raise
        except KeyboardInterrupt:
            import matplotlib.pyplot as plt
            plt.plot(self.plotX1, self.plotY1, 'b.-')
            plt.plot(self.plotX2, self.plotY2, 'r.-')
            plt.show()

    def positionHandler(self, channel, data):
        rcvMsg = lcm_velocity_t.decode(data)
        if rcvMsg.cycle_start:
            self.cycleStartUtime = rcvMsg.utime

        if time.time() >= self.timeLastSent + self.timestep:
            referencePoint = self.referencePointFn(rcvMsg.utime - self.cycleStartUtime)
            vCommand = self.controllerFn(referencePoint - rcvMsg.position_m[1])
            sendMsg = lcm_velocity_t()
            sendMsg.utime = long(time.time() * 1e6)
            sendMsg.command_v_mps = vCommand
            self.lcmObj.publish("V_COMMAND", sendMsg.encode())
            self.timeLastSent = time.time()
            self.plotX2.append(usToS(rcvMsg.utime))
            self.plotY2.append(referencePoint)

        if DEBUG_HANDLE:
            print("Received message on channel \"%s\"" % channel)
            print("\tutime         = %s" % str(rcvMsg.utime))
            print("\tcommand_v_mps = %s" % str(rcvMsg.command_v_mps))
            print("\tposition      = %s" % str(rcvMsg.position_m))
            print("\tcycle_start   = %s" % str(rcvMsg.cycle_start))
        self.plotX1.append(usToS(rcvMsg.utime))
        self.plotY1.append(rcvMsg.position_m[1])


def constantFn(dUT):
    '''
    input (dUT): delta utime from the start of the system
    returns: referencePoint of the system in meters
    '''
    return CONSTANT_FN_VALUE_M


def sineFn(dUT):
    '''
    input (dUT): delta utime from the start of the system
    returns: referencePoint of the system in meters
    '''
    t = usToS(dUT)
    return (np.sin((SINE_FREQ_RPS * t) + SINE_ANGULAR_OFFSET_R) * SINE_AMPLITUDE_M) + SINE_OFFSET_M


def sawtoothFn(dUT):
    '''
    input (dUT): delta utime from the start of the system
    returns: referencePoint of the system in meters
    '''
    t = usToS(dUT)
    return (SAWTOOTH_SLOPE * t) % SAWTOOTH_Y_LIMIT_M


def squareFn(dUT):
    '''
    input (dUT): delta utime from the start of the system
    returns: referencePoint of the system in meters
    '''
    t = usToS(dUT)
    return (((t / SQUARE_WIDTH_S) % 2) > 1) * SQUARE_HEIGHT_M


def circleFn(dUT):
    '''
    input (dUT): delta utime from the start of the system
    returns: referencePoint of the system in meters
    ellipse equation = x^2/x_rad^2 + y^2/y_rad^2 = 1
    y = sqrt((1 - x^2/x_rad^2) * y_rad^2)
    '''
    t = usToS(dUT)
    x = (t % (2 * CIRCLE_RADIUS_S)) - CIRCLE_RADIUS_S
    y = np.sqrt((1 - (pow(x, 2) / pow(CIRCLE_RADIUS_S, 2))) * pow(CIRCLE_RADIUS_M, 2))
    invert = ((((t / (2 * CIRCLE_RADIUS_S)) % 2) > 1) * 2 - 1)
    if CIRCLE_INVERT:
        return y * invert
    else:
        return y


def pCtrlr(eM):
    '''
    input (eM): error in meters
    returns: velocity setpoint of reacting system (m/s)
    '''
    return P_CONSTANT * eM


def usToS(utime):
    ''' Converts values in microseconds to values in seconds '''
    return (utime / 1.0e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts a 1D controller",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--controller-function",
                        help="Choose which function to use as a controller",
                        action="store",
                        default="proportional")
    parser.add_argument("-s", "--reference-point-function",
                        help="Choose which function to use as a referencePoint",
                        action="store",
                        default="constant")
    parser.add_argument("-t", "--timestep-s",
                        help="Respond to position data every -t seconds",
                        type=float,
                        default=0.01)
    args = parser.parse_args()

    if args.reference_point_function == "constant":
        referencePointFn = constantFn
    elif args.reference_point_function == "sine":
        referencePointFn = sineFn
    elif args.reference_point_function == "saw":
        referencePointFn = sawtoothFn
    elif args.reference_point_function == "square":
        referencePointFn = squareFn
    elif args.reference_point_function == "circle":
        referencePointFn = circleFn
    else:
        raise ValueError("Didn't recognize referencePointFn {}".format(args.reference_point_function))

    if args.controller_function == "proportional":
        controllerFn = pCtrlr
    else:
        raise ValueError("Didn't recognize controllerFn".format(args.controller_function))

    C1D = Controller1D(referencePointFn, controllerFn, args.timestep_s)
