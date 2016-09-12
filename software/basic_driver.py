#!/usr/bin/env python

import serial
import time
import numpy as np

import lcm

from lcmtypes import lcm_velocity_t


Y_V_LIMIT = 0.5


class VelocityDriver1D():
    def __init__(self, timeStep, xVelocity, xLimit):
        self.timeStep = timeStep
        self.xVelocity = xVelocity
        self.xLimit = xLimit
        self.xPosition = 0.0
        self.yPosition = 0.0

        self.lcmObj = lcm.LCM()

        # Open grbl serial port
        self.serialConnection = serial.Serial('/dev/ttyUSB0', 115200)

        # Wake up grbl
        self.serialConnection.write("\r\n\r\n")
        print("Wake up grbl")
        time.sleep(2)   # Wait for grbl to initialize 
        print("Woken up!")
        self.serialConnection.flushInput()  # Flush startup text in serial input

        # GCODE list: http://www.cnccookbook.com/CCCNCGCodeRef.html
        lines = ["G21",  # Program coordinates are in mm
                 "G91",  # Incremental programming of XYZ (command 5 does +5 instead of global position 5)
                 "G94",  # Feed mode is units/minute. Because of G21, this means mm/minute
                 "G01 F1500",  # Move in a straight line, feedrate 1500 (mm/min as defined by G21/G94)
                 "$$"]  # Display the settings
        self.sendLines(lines)

    def startRunning(self):
        while xPosition < xLimit:
            if time.now() > (timeLastSent + self.timeStep):
                pass
            else:
                self.lcmObj.handle()
            time.sleep(0.001)

    def cleanUp(self, giveRetreatOption=True):
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

        # Close file and serial port
        self.serialConnection.close()

    def sendLines(self, lines):
        # Stream g-code to grbl
        for line in lines:
            lineStrip = line.strip() # Strip all EOL characters for consistency
            # For now, I'm referring to X and Y swapped from what the XPro thinks
            #   of it as. This corrects that
            lineCorrected = self.swapXandY(lineStrip)

            print("Sending: " + lineCorrected)
            self.serialConnection.write(lineCorrected + "\n") # Send g-code block to grbl
            print("Sent!")

            if lineCorrected == "$$":
                for idx in range(31):
                    grbl_out = self.serialConnection.readline().strip()
                    print("grbl_out: {}".format(grbl_out))
            else:
                grbl_out = self.serialConnection.readline(30) # Wait for grbl response with carriage return
                print("grbl_out: {}".format(grbl_out))

    def handleCommands(self, channel, data):
        try:
            self.commandV = lcm_velocity_t.decode(data).command_v_mps
        except:
            self.cleanUp(giveRetreatOption=False)
            raise


def swapXandY(line):
    xIdx = line.index('X')
    yIdx = line.index('Y')
    line = line[:xIdx] + 'Y' + line[xIdx + 1:]
    line = line[:yIdx] + 'X' + line[yIdx + 1:]
    return line
