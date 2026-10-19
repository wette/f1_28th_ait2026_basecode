from localization.helper_functions import *
import math

import time

from tools.OverHeadView import OverHeadView
from tools.simulation import Simulation


####################################
# IMPORTANT NOTICE:
#
# If the Script ends with a segfault, it is caused by FWGL and tensorflow not beeing happy when used together on Ubuntu 24.04.
# To run the script anyways, you need to set the env variable __GLX_VENDOR_LIBRARY_NAME to nvidia.
# 
# source .venv/bin/activate
# export __GLX_VENDOR_LIBRARY_NAME=nvidia
# python simulation_singlethreaded.py
#
# You can also add export __GLX_VENDOR_LIBRARY_NAME=nvidia to your bashrc.
#
#####################################



#map parameters (realLifeRennstrecke.png resolution)
vertical_resolution_px = 1200
horizontal_resolution_px = 1920

#ego camera properties
frames_per_seconds = 30
ego_cam_opening_angle_horizontal_degrees = 55.0
ego_cam_resolution = (320, 180)
height_over_ground_camera_m = 0.055    #how high is the camera mounted over ground?
ego_cam_pitch_degrees = -5 #camera pitch in degrees (0 = parallel to ground, negative = looking downwards)

#physical parameters of camera and vehicle features
meters_to_pixels = 681     #how many pixels are in one meter?


def checkFinishLine(v, current_frame, top_left:tuple, bottom_right:tuple):
    #debug finishline:
    cv.line(current_frame, top_left, (top_left[0], bottom_right[1]), (0,255,255), 3)
    
    currently_on_finishline = False
    if top_left[0] < v.x < bottom_right[0] and \
        top_left[1] < v.y < bottom_right[1]:
        currently_on_finishline = True

    if not v.is_on_finish_line and currently_on_finishline:
        #just passed the line:
        v.is_on_finish_line = True
        if v.time_start_of_lap > 0:
            v.last_laptime = time.time() - v.time_start_of_lap
        v.time_start_of_lap = time.time()

    if v.is_on_finish_line and not currently_on_finishline:
        v.is_on_finish_line = False

    return current_frame


def main():
    vehicle_starting_position = (850, 1030, math.radians(0.0)) # x_px,y_px,yaw_rad

    #setup simulation
    sim = Simulation(track_border_filename="track_borders_simulation.npy",
                     vehicle_color="yellow",
                     initial_pose_x_y_yaw=vehicle_starting_position,
                     meters_to_pixels=meters_to_pixels,
                     vertical_resolution_px=vertical_resolution_px,
                     height_over_ground_camera_m=height_over_ground_camera_m,
                     dt = 1.0/frames_per_seconds,
                     time_limit_s=60,
                     track_png_path="realLifeRennstrecke.png",
                     track_size_px=(horizontal_resolution_px, vertical_resolution_px),
                     egocam_render_size=ego_cam_resolution,
                     egocam_fov_y_deg=ego_cam_opening_angle_horizontal_degrees,
                     egocam_pitch_deg=ego_cam_pitch_degrees,
                     egocam_skybox_seed=4)

    #setup overhead view:
    ohv = OverHeadView(backgroundImageFilename="realLifeRennstrecke.png",
                       track=sim.racetrack)

    target_loop_execution_time = 1.0/frames_per_seconds
    loop_start_time = time.time()

    while True:
        #reset simulation for next run.
        simdata = sim.reset()
        steer_cmd = 0.0

        while not simdata["simulation_over"] and sim.v is not None: #until the vehicle went offroad, or timelimit

            # read current values from simulation:
            obs_bgr = simdata["camera_image_bgr"]
            cur_steer = steer_cmd
            cur_speed = simdata["vehicle_speed_mps"]

            # compute steering and velocity according to the current state of the vehicle and the track using a controller
            desired_speed_mps, steering_angle_rad, _,_,_ = sim.v.compute_next_command(1.0/frames_per_seconds, [])

            #limit steering command to the physical limits of the vehicle
            steering_angle_rad = np.clip(steering_angle_rad, math.radians(sim.v.min_steering_angle_deg), math.radians(sim.v.max_steering_angle_deg))

            # check if steering needs to be inverted for this vehicle
            if sim.v.steering_inverted:
                steering_angle_rad = -steering_angle_rad

            # simulate what happens in the next timestep with the given steering and velocity commands
            simdata = sim.nextStep(target_velocity_mps=desired_speed_mps,
                                target_steering_angle_rad=steering_angle_rad)


            # get overhead view image:
            overhead_view = ohv.getImage(sim.v)

            # debug: check the start/finishline to display a laptime
            checkFinishLine(sim.v, overhead_view, top_left=(1000, 950), bottom_right=(1050, 1120))
            cv.putText(overhead_view, f"Lap time: {(time.time()-sim.v.time_start_of_lap):.2f}s ({sim.v.last_laptime:.2f}s)", (800, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                                    hueToBGR(120/2), 2, cv.LINE_AA)


            # debug: show ego camera image
            cv.imshow("Ego-camera OpenGL", simdata["camera_image_bgr"])
            #debug: show overhead image:
            small_overhead_view = cv.resize(overhead_view, (0,0), fx=0.5, fy=0.5) 
            cv.imshow("Overhead View", small_overhead_view)

            # display images
            cv.waitKey(1)


            # measure the actual execution time of the loop
            actual_loop_execution_time = time.time() - loop_start_time
            loop_start_time = time.time()

            #debug: real time simulation
            if actual_loop_execution_time < target_loop_execution_time:
                #print(f"Loop execution time: {actual_loop_execution_time:.4f}s, sleeping for {target_loop_execution_time - actual_loop_execution_time:.4f}s to maintain real-time simulation.")
                time.sleep(target_loop_execution_time - actual_loop_execution_time)


            #end: while not simdata["simulation_over"]






if __name__ == "__main__":
    main()