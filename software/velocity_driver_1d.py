#!/usr/bin/env python

import numpy as np
import select
import serial
import signal
import time

import lcm

from lcmtypes import lcm_velocity_t


Y_V_LIMIT_MPS = 0.01  # In m/s (600 mm/min)


class VelocityDriver1D():
    def __init__(self, timeStep, xVelocity, xLimit):
        self.timeStep = timeStep                     # Time in s
        self.xVelocity = xVelocity                   # Velocities in m/s
        self.xStep = self.xVelocity * self.timeStep  # Position in m
        self.yVelocity = 0.0                         # Velocities in m/s
        self.xLimit = xLimit                         # Position in m
        self.xPosition = 0.0                         # Position in m
        self.yPosition = 0.0                         # Position in m
        self.loopTimeout = 0.005                     # Time in s
        self.running = False

        self.lcmObj = lcm.LCM()
        self.subscription = self.lcmObj.subscribe("V_COMMAND", self.handleCommands)

        # Open grbl serial port
        self.serialConnection = serial.Serial('/dev/ttyUSB0', 115200)

        # Wake up grbl
        self.serialConnection.write("\r\n\r\n")
        print("Wake up grbl")
        time.sleep(2)   # Wait for grbl to initialize 
        print("Woken up!")
        self.serialConnection.flushInput()  # Flush startup text in serial input

        # GCODE list: http://www.cnccookbook.com/CCCNCGCodeRef.html
        lines = ["G21",        # Sets program coordinates to be in mm
                 "G91",        # Sets incremental programming of XYZ (now command 5 does +5 instead of global position 5)
                 "G94",        # Now feed mode is units/minute. Because of G21, this means mm/minute
                 "G01 F1500",  # Sets movement in a straight line, feedrate 1500 (mm/min as defined by G21/G94)
                 "$$"]         # Display the settings
        self.sendLines(lines)

    def startRunning(self):
        '''
        Starts the event loop that handles incoming commands and takes position
        steps at the appropriate timesteps
        '''
        # Each time we call startRunning, set running to True for the while
        #   loop and False after that
        self.running = True
        timeLastSent = 0
        xPositionAtLastSend = self.xPosition
        yPositionAtLastSend = self.yPosition
        yStep = 0.0

        while self.xPosition <= self.xLimit and self.running:
            if time.time() > (timeLastSent + self.timeStep):
                self.xPosition = xPositionAtLastSend + self.xStep
                self.yPosition = yPositionAtLastSend + yStep
                xPositionAtLastSend = self.xPosition

                yStep = self.yVelocity * self.timeStep
                vVector = np.array([self.xVelocity, self.yVelocity])
                feedrateMPS = np.linalg.norm(vVector)
                line = "X{} Y{} F{}".format(mToMM(self.xStep),
                                            mToMM(yStep),
                                            mpsToMMPMin(feedrateMPS))
                print('Sending combined step command')
                print('\tConstant xStep = {}m, yStep = {}m'.format(self.xStep, yStep))
                print('\txPosition: {}'.format(self.xPosition))
                self.sendLines([line])
                timeLastSent = time.time()
            else:
                self.xPosition = xPositionAtLastSend + \
                                 (time.time() - timeLastSent) * self.xVelocity
                self.yPosition = yPositionAtLastSend + \
                                 (time.time() - timeLastSent) * self.yVelocity

                # Wait for timeout to handle lcmObj, otherwise just pass
                rfds, wfds, efds = select.select([self.lcmObj.fileno()],
                                                 [], [], self.loopTimeout)
                if rfds:
                    self.lcmObj.handle()
            time.sleep(0.001)

        print("Out of the loop! Cleaning up...")
        self.running = False
        self.cleanUp(giveRetreatOption=True)

    def cleanUp(self, giveRetreatOption=True):
        self.running = False
        self.sendLines(["X0 Y0"])

        if giveRetreatOption:
            # Wait here until grbl is finished to close serial port and file.
            shouldRetreat = raw_input("Program done. Do you want to retreat" +\
                                      " to the starting position? (y/n)")
            if shouldRetreat.lower() == 'y':
                retreatCorrect = raw_input("It looks like the head has moved" +\
                                           " ({}, {})m".format(self.xPosition,
                                                               self.yPosition) +\
                                           " from the beginning, undo? (y/n)")
                if retreatCorrect.lower() == 'y':
                    lines = ["X{} Y{}".format(int(-self.xPosition),
                                              int(-self.yPosition))]
                    self.sendLines(lines)
            print("Quitting")
        else:
            print("Instructed not to retreat, quitting")

        # Close file and serial port
        self.serialConnection.close()

    def sendLines(self, lines):
        if self.serialConnection.closed:
            raise RuntimeError("Told to send commands but serial port closed")

        # Stream g-code to grbl
        for line in lines:
            lineStrip = line.strip() # Strip all EOL characters for consistency
            if 'X' in lineStrip.upper() and 'Y' in lineStrip.upper():
                # For now, I'm referring to X and Y swapped from what the XPro
                #   thinks of it as. This corrects that if we are sending
                #   strings like "X10 Y20 F1500s"
                lineCorrected = swapXandY(lineStrip)
            else:
                lineCorrected = lineStrip

            print("Sending: " + lineCorrected)
            # Send g-code block to grbl
            self.serialConnection.write(lineCorrected + "\n")
            print("Sent!")

            if lineCorrected == "$$":
                for idx in range(31):
                    grbl_out = self.serialConnection.readline().strip()
                    print("grbl_out: {}".format(grbl_out))
            else:
                # Wait for grbl response with carriage return
                grbl_out = self.serialConnection.readline(30)
                print("grbl_out: {}".format(grbl_out))

    def handleCommands(self, channel, data):
        try:
            yCommand = lcm_velocity_t.decode(data).command_v_mps
            if abs(yCommand) <= Y_V_LIMIT:
                self.yVelocity = yCommand
            else:
                self.yVelocity = np.sign(yCommand) * Y_V_LIMIT
        except:
            self.cleanUp(giveRetreatOption=False)
            raise


def swapXandY(line):
    xIdx = line.index('X')
    yIdx = line.index('Y')
    line = line[:xIdx] + 'Y' + line[xIdx + 1:]
    line = line[:yIdx] + 'X' + line[yIdx + 1:]
    return line


def mToMM(value):
    ''' Converts values in m (float) to mm (int). Works for m and m/s '''
    return int(round(value * 1e3))


def mmToM(value):
    ''' Converts values in mm (int) to m (float). Works for mm and mm/s '''
    return value / 1.0e3


def mpsToMMPMin(value):
    ''' Converts values in m/s (float) to mm/min (int) '''
    return int(round(value * 1e3 * 60))
