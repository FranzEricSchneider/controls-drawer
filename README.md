README

** HOW TO RUN
In terminal 1
python controller_1d.py --timestep-s 0.01 --controller-function proportional --reference-point-function constant

In terminal 2 - run every time you want the thing to run
python velocity_driver_1d.py --timestep-s 0.05 --x-velocity 0.01 --x-limit 0.1