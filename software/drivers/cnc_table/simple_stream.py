#!/usr/bin/env python
"""
---------------------
The MIT License (MIT)

Copyright (c) 2012 Sungeun K. Jeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
---------------------
"""

from random import random
import serial
import time
import numpy as np

# Open grbl serial port
s = serial.Serial('/dev/ttyUSB0', 115200)

# Wake up grbl
s.write("\r\n\r\n")
print("Wake up grbl")
time.sleep(2)   # Wait for grbl to initialize 
print("Woken up!")
s.flushInput()  # Flush startup text in serial input

cmd = []

# GCODE list: http://www.cnccookbook.com/CCCNCGCodeRef.html
lines = ["G21",  # Program coordinates are in mm
         "G91",  # Incremental programming of XYZ (command 5 does +5 instead of global position 5)
         "G94",  # Feed mode is units/minute. Because of G21, this means mm/minute
         "G01 F1500",  # Move in a straight line, feedrate 1500 (mm/min as defined by G21/G94)
         # "$$"]  # Display the settings
         "$$",  # Display the settings
         "G01 F1500",  # Move in a straight line, feedrate 1500 (mm/min as defined by G21/G94)
         "X0.5 Y0.5"]  # Because of G01, this should move X and Y by their units

# These do different things according to what is commented out
# scalar = 8
# for i in range(296 / scalar):
#     # speed = ((random() - 0.5) * 2) * 100 + 1500
#     # lines += ["G01 F{0:.1f}".format(speed), "X{} Y0".format(scalar)]
#     lines += ["X{} Y0".format(scalar)]

# ## These do different things according to what is commented out
# maxX = 1200
# X = range(0, maxX, 100)
# negative = False
# for i in X:
#     # # Version 1: Leave the speed constant and do a 90 degree turn
#     # v = float(i) / 50
#     # lines.append("X{0:.2f} Y{1:.2f}".format((maxX / 50.0) - v, v))
#     # Version 2: Leave the step constant and sweep the speed from 250 to 350
#     v = 250 + i
#     lines.append("G01 F{}".format(v))
#     lines.append("X{}15 Y15".format('-' if negative else ''))
#     negative = not negative
#     # # Version 3: Do a sine wave with a normalized vector
#     # radians = i * np.pi / 180.0
#     # xStep = np.cos(radians)
#     # scalar = 2 / 8.0
#     # # We want to always step a constant Y value
#     # lines.append("X{0:.2f} Y{1:.2f}".format(xStep * scalar, 0.25))

import ipdb
ipdb.set_trace()

# Stream g-code to grbl
for line in lines:
    l = line.strip() # Strip all EOL characters for consistency
    print "Sending: " + l,
    s.write(l + "\n") # Send g-code block to grbl
    print("Sent!")
    if l == "$$":
        for i in range(31):
            grbl_out = s.readline().strip()
            print("grbl_out: {}".format(grbl_out))
    else:
        grbl_out = s.readline(30) # Wait for grbl response with carriage return
        print("grbl_out: {}".format(grbl_out))
    ## If the sleep is commented out the controller will remember a fixed number
    ##   of commands and then will buffer the rest
    # time.sleep(0.1)

# Wait here until grbl is finished to close serial port and file.
raw_input("  Press <Enter> to exit and disable grbl.") 

# Close file and serial port
s.close()