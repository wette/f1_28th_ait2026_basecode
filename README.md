# F1/28th Control Software
Autonomous driving Software using a top-down view camera to control racecars in 1:28th scale.

## Installation
Tested with Python 3.11.

To install all dependencies in a virtual environment, execute the following commands:

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt

## Vehicle Configuration
See `vehicle_config.py`

The color of the vehicle is denoted by the color of the marker on the roof of the vehicle. If new colors are introduced, they need to be named in the 
ColorMap located in `localization/camera.py` and associated with the correct HSV Hue value.

## Control the actual Vehicles
Turn the vehicles on, place them on the racetrack and run `main.py`

## Simulation Environment
To run the Software without physical hardware, run `simulation_singlethreaded.py`. The script uses a very simple motion model to simulate movement.
The script can be used to test out new algorithms or generate training data for machine learning.



`simulation.py` is a multi-threaded simulation for debugging purposes, only.
