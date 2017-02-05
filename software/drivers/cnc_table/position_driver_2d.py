#!/usr/bin/env python

import argparse
import numpy as np
import serial
import time

import lcm

from utils import grbl_tools
from utils import lcm_msgs


DEBUG_HANDLE = False
DEBUG_SENDLINES = False
LOOP_TIMEOUT = 0.005  # Time in s
LOOP_SLEEP = 0.005    # Time in s


class positionDriver2D():
    def __init__(self, velocityLimit):
        self.velocityLimit = velocityLimit    # Velocities in m/s
        self.position = np.array([0.0, 0.0])  # Position in m
        self.running = False

        # Used to keep track of commands that need to be/were executed
        self.inMessages = []
        self.sentMessages = []

        # Set up LCM objects
        self.lcmobj = lcm.LCM()
        self.inputChannel = "POSITION_COMMAND"
        self.outputChannel = "TOOL_STATE"
        self.sub = self.lcmobj.subscribe(self.inputChannel, self.handleCommands)

        # Open grbl serial port
        self.serial = serial.Serial("/dev/ttyUSB0", 115200)

        # Wake up grbl (includes a sleep for several seconds)
        grbl_tools.openSerial(self.serial)

        # Set up the table with the normal settings (mm/min, relative
        #   positioning, etc.)
        lines = grbl_tools.setupBasicSystem()
        grbl_tools.sendLines(self.serial, lines, DEBUG_SENDLINES)

    def startRunning(self):
        '''
        Starts the event loop that handles incoming commands and takes position
        steps at the appropriate timesteps
        '''

        # Each time we call startRunning, set running to True for the while
        #   loop and False after that
        self.running = True
        timeLastSent = time.time()
        vectorLastSent = np.array([0.0, 0.0])
        firstLoopMsg = True

        while self.running:
            now = time.time()

            # Update estimation of state
            try:
                positionCmds = np.array([cmd.position for cmd in self.sentMessages])
                distTravelled = (now - timeLastSent) * self.sentMessages[-1].velocity
                if distTravelled < np.linalg.norm(self.sentMessages[-1].position):
                    # If we have sent commands W, X, Y, Z, this computes the tool
                    #   position at the end of command Y. Theoretically, command Z is
                    #   happening right now
                    lastCmdPosition = np.sum(positionCmds[:-1], axis=0)
                    self.position = lastCmdPosition + vectorLastSent * distTravelled
                else:
                    # We should have stopped moving at this point (roughly), the
                    #   (x, y) position should be the sum of all the previous
                    #   movements
                    self.position = np.sum(positionCmds, axis=0)

                # Send out state
                stateMsg = lcm_msgs.auto_instantiate(self.outputChannel)
                stateMsg.position = list(self.position)
                stateMsg.cycle_start = firstLoopMsg
                firstLoopMsg = False
                self.lcmobj.publish(self.outputChannel, stateMsg.encode())
            except IndexError:
                # An array hasn't been populated yet
                pass

            # Send out a new command if it is relevant
            if len(self.sentMessages) > 0:
                now = time.time()
                distanceToTravel = np.linalg.norm(self.sentMessages[-1].position)
                timeToTravel = distanceToTravel / self.sentMessages[-1].velocity
            else:
                timeToTravel = 0.0
            try:
                if now >= (timeLastSent + timeToTravel):
                    line = grbl_tools.makeCmd(self.inMessages[0].position[0],
                                              self.inMessages[0].position[1],
                                              self.inMessages[0].velocity)
                    grbl_tools.sendLines(self.serial, [line], DEBUG_SENDLINES)
                    self.sentMessages.append(self.inMessages.pop(0))

                    # For bookkeeping, keep the time and the unit vector of the
                    #   last sent position command
                    timeLastSent = now
                    vectorLastSent = np.array(self.sentMessages[-1].position)
                    vectorLastSent /= vectorLastSent
            except IndexError:
                # An array hasn't been populated yet
                pass

            # Handle incoming requests if they've come in. If not, timeout and
            #   move on with your life
            lcm_msgs.lcmobj_handle_msg(self.lcmobj, LOOP_TIMEOUT)

            # Sleep in the loop
            time.sleep(LOOP_SLEEP)

        print("Out of the main loop! Cleaning up...")
        self.running = False
        self.cleanUp(offerRetreat=True)

    def handleCommands(self, channel, data):
        """
        If you get a command message, limit the velocity and save it to be sent
        to the table later
        """
        try:
            msg = lcm_msgs.auto_decode(channel, data)
            msg.velocity = max(min(msg.velocity, self.velocityLimit), 0.0)
            if DEBUG_HANDLE:
                print("Received msg:\n\tutime: {}\n\tvelocity: {}\n\tposition:"
                      " {}".format(msg.utime, msg.velocity, msg.position))
            self.inMessages.append(msg)
        except:
            self.cleanUp(offerRetreat=False)
            raise

    def cleanUp(self, offerRetreat=True):
        # Send a halt message
        self.running = False
        grbl_tools.sendLines(self.serial, grbl_tools.haltMessage(), DEBUG_SENDLINES)

        # Retreat the head to the starting position if commanded
        grbl_tools.retreat(offerRetreat, self.serial, self.position)

        # Close serial port to the CNC table
        self.serial.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts a 2D position driver",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 0.01 m/s is quivalent to 600 mm/min
    parser.add_argument("-v", "--velocity-limit",
                        help="Velocity (m/s) to limit vector velocity at",
                        type=float,
                        default=0.01)
    args = parser.parse_args()

    PD2D = positionDriver2D(args.velocity_limit)
    PD2D.startRunning()
