from localization.camera import Camera
from localization.vehicle import Vehicle
from localization.helper_functions import hueToBGR, createPlot
from localization.pid_controller import PIDController
from control.track import Track
from control.disparity_extender import DisparityExtender
from control.lidarSensor import LidarSensor
from shapely import LineString
import numpy as np
import math

from multiprocessing import Process, Manager
import cv2 as cv
import time
import copy
import collections


#ip addresses and parameters of vehicles
from vehicle_config import vehicles

#camera parameters
vertical_resolution_px = 1200
horizontal_resolution_px = 1920
frames_per_seconds = 90
opening_angle_vertical_degrees = 88.0
opening_angle_horizontal_degrees = 126.0

#physical parameters of camera and vehicle features
meters_to_pixels = 683     #how many pixels are in one meter?
max_speed_vehicle_mps = 5.0        #max speed of a car in meters per second
minimum_brightness = 1.0 #2.7   #used to brighten the image of the webcam
threshold_brightness_of_black = 150       #rgb from 0-255
threshold_brightness_of_white = 200        #rgb from 0-255
circle_diameter_meters = 0.025  #diameter of black and white dots (2cm)
size_between_black_and_white_center_meters = 0.08  #8cm from center to center
height_over_ground_black_meters = 0.055    #how high is the black dot on the vehicle measured from the ground
height_over_ground_white_meters = 0.035    #how high is the white dot on the vehicle measured from the ground

#camera correction parameters
cameraMatrix = np.array([[1.19164513e+03, 0.00000000e+00, 9.32255365e+02],
                         [0.00000000e+00, 1.19269246e+03, 5.44789222e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distortionCoefficients = np.array([[ 0.02473071, -0.39668063,  0.00151336,  0.00085757,  0.25759047]])


def signed_radius(startpos, endpos, startHeading):
    x0, y0 = startpos
    x1, y1 = endpos
    psi0 = startHeading  # rad

    dx = x1 - x0
    dy = y1 - y0

    # Linksnormale zur Startrichtung
    nx = -math.sin(psi0)
    ny =  math.cos(psi0)

    d_perp = dx * nx + dy * ny
    if abs(d_perp) < 1e-12:
        return float("inf")  # (nahezu) geradeaus, Radius -> unendlich

    R = (dx*dx + dy*dy) / (2.0 * d_perp)  # Vorzeichen: + links, - rechts (bei dieser Konvention)
    return R

def lenkwinkel_from_radius(radius_m, radstand_m):
    delta_rad = math.atan(radstand_m / radius_m)
    delta_deg = math.degrees(delta_rad)
    return delta_deg


def main():

    #setup camera
    cam = Camera(vertical_resolution_px = vertical_resolution_px,
                 horizontal_resolution_px = horizontal_resolution_px,
                 frames_per_seconds = frames_per_seconds,
                 opening_angle_vertical_degrees = opening_angle_vertical_degrees,
                 opening_angle_horizontal_degrees = opening_angle_horizontal_degrees,
                 meters_to_pixels = meters_to_pixels,
                 max_speed_vehicle_mps = max_speed_vehicle_mps,
                 minimum_brightness = minimum_brightness,
                 threshold_brightness_of_black = threshold_brightness_of_black,
                 threshold_brightness_of_white = threshold_brightness_of_white,
                 circle_diameter_meters = circle_diameter_meters,
                 size_between_black_and_white_center_meters = size_between_black_and_white_center_meters,
                 height_over_ground_black_meters = height_over_ground_black_meters,
                 height_over_ground_white_meters = height_over_ground_white_meters,
                 cameraMatrix = cameraMatrix,
                 distortionCoefficients = distortionCoefficients,
                 from_file=None,
                 create_debug_video=False)
    
    racetrack = Track()
    racetrack.loadFromFile("track_borders.npy")


    while cam.is_video_stream_active():
        if cam.detectVehicles() > 0:
            print(f"found {len(cam.tracked_vehicles)} vehicles.")

            #setup vehicles
            for v in cam.tracked_vehicles:
                if v.color in vehicles.keys():
                    print("Setting up vehicle with color: ", v.color)
                    #set physical properties
                    v.setPhysicalProperties(vehicles[v.color]["length_m"], 
                                            vehicles[v.color]["width_m"], 
                                            vehicles[v.color]["rear_axle_offset_m"], 
                                            vehicles[v.color]["steering_measurements"], 
                                            0,
                                            0,
                                            vehicles[v.color]["speed_factor"],
                                            vehicles[v.color]["min_steering_angle"],
                                            vehicles[v.color]["max_steering_angle"],
                                            vehicles[v.color]["steering_inverted"],
                                            vehicles[v.color]["height_over_ground_colored_marker_m"],
                                            vehicles[v.color]["height_over_ground_white_marker_m"],
                                            vehicles[v.color]["delay_ms"])
                    
                    #setup motor parameters
                    v.initMotorPID(vehicles[v.color]["motor_pid"][0],
                                   vehicles[v.color]["motor_pid"][1],
                                   vehicles[v.color]["motor_pid"][2],
                                   error_history_length=50)
                    
                    #setup IP communication
                    v.initNetworkConnection(ip_adress=vehicles[v.color]["ip"], port=vehicles[v.color]["port"])
                    
                    #setup controller
                    v.controller = None
                    
                    #setup lidar
                    v.lidar = None
                    
                else:
                    print("Could not properly detect color of vehicle!")

            #tracking loop
            delta_t = 1.0/frames_per_seconds

            last_box = None
            x = 0
            while len(cam.tracked_vehicles) > 0:

                steering_values = []
                startpos_of_vehicle = None
                endpos_of_vehicle = None
                startheading_of_vehicle = None

                in_trajectory_recording = False

                
                results = []
                for target_steering_angle_pwm in [10, 50, 90, 130, 180]:
                    calibrated = False

                    print(f"Testing for steering angle with Servo PWM {target_steering_angle_pwm}")
                    print("Move Vehicle back to start. Then press 's' to start the measurement.")
                    print("Then, slowly move the vehicle forwards for roughly 50cm, or if the vehicle did quarter of a circle.")
                    print("Then, press 'e' to end the measurement.")

                    while not calibrated:

                        
                        
                        #update position of each vehicle
                        cam.trackVehicles()

                        #frame for display
                        frame = cam.get_last_frame()


                        for vehicle in cam.tracked_vehicles:
                            vehicle.ttl = 15 #do not let the cam decrement the ttl of the vehicle, since we are in a calibration mode and want to keep the vehicle in view at all times.
                            try:
                                if vehicle.color in vehicles.keys():
                                    #send steering command to vehicle:
                                    output = [0, 0, target_steering_angle_pwm]
                                    msg = ";".join(f"{e:03}" for e in output)
                                    vehicle.sock.sendto(bytes(msg, "utf-8"), (vehicle.IP, vehicle.port))

                                    #draw outcome.
                                    boundingbox = vehicle.getBoundingBox(cam.current_time)
                                    color = (0, 255, 0)
                                    if vehicle.ttl < 15:
                                        color = (0, 0, 255)
                                    Camera.drawBoundingBox(frame, boundingbox, color=color)
                                    
                            except:
                                pass
                            
                        cv.imshow('frame', frame)
                        key = cv.waitKey(1)
                        if key == ord('q'):
                            break

                        if len(cam.tracked_vehicles) > 0:
                            vehicle = cam.tracked_vehicles[0]

                            if key == ord('s'):
                                startpos_of_vehicle = vehicle.getPosition()
                                endpos_of_vehicle = None
                                startheading_of_vehicle = vehicle.getOrientation()
                                in_trajectory_recording = True
                                print(f"Start Measurement for {vehicle.color} vehicle...")

                            if key == ord('e'):
                                endpos_of_vehicle = vehicle.getPosition()
                                in_trajectory_recording = False

                                radius = signed_radius(startpos_of_vehicle, endpos_of_vehicle, startheading_of_vehicle) / cam.meters_to_pixels
                                lw_deg = lenkwinkel_from_radius(radius_m=radius, radstand_m=0.095)

                                print(f"PWM: {target_steering_angle_pwm} results in steering angle {lw_deg*-1} [deg]")

                                results.append( (target_steering_angle_pwm, lw_deg*-1) )
                                calibrated = True
                print("Calibration Results:")
                print(results)


            print("No more cam.tracked_vehicles left.")


        else:
            print("Could not detect any vehicle - restarting.")


if __name__ == "__main__":
    main()
