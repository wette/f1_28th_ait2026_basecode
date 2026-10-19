from control.track import Track
from localization.vehicle import Vehicle
from vehicle_config import vehicles
from control.disparity_extender import DisparityExtender
from control.lidarSensor import LidarSensor
from tools.TrackProgressGated import TrackProgressGated
import math
from shapely.geometry import LinearRing, Polygon, box
import shapely

from localization.helper_functions import *

from tools.renderer3d import openGL3d
    

class Simulation:
    def __init__(self,  track_border_filename="track_borders_simulation.npy", 
                        vehicle_color="green",
                        initial_pose_x_y_yaw=(650, 500, math.radians(-30.0)),
                        meters_to_pixels=681,
                        vertical_resolution_px=1200, 
                        height_over_ground_camera_m=0.055,
                        dt=1/30,
                        time_limit_s=60,
                        track_png_path="realLifeRennstrecke.png",
                        track_size_px=(1920, 1200),
                        egocam_render_size=(320, 180),
                        egocam_fov_y_deg=80.0,
                        egocam_fx=None,
                        egocam_fy=None,
                        egocam_cx=None,
                        egocam_cy=None,
                        egocam_distortion=None,
                        egocam_skybox_seed=4,
                        egocam_pitch_deg=0,
                        egocam_forward_offset_m=0.0
                 ):
        
        self.vertical_resolution_px=vertical_resolution_px 
        self.height_over_ground_camera_m = height_over_ground_camera_m
        self.dt = dt
        self.time_limit_s = time_limit_s
        self.egocam_pitch_rad = np.deg2rad(egocam_pitch_deg)
        
        #load track outlines.
        self.racetrack = Track()
        self.racetrack.loadFromFile(track_border_filename)

        self.progressCalculator = TrackProgressGated(left_ring=LinearRing(self.racetrack.innerBounds),
                                                     right_ring=LinearRing(self.racetrack.outerBounds),
                                                     start_xy=initial_pose_x_y_yaw[0:2],
                                                     n_samples=1000)

        self.initial_pose_x_y_yaw = initial_pose_x_y_yaw

        self.v = None
        self.current_motor_value = 0
        self.last_progress = 0
        self.current_time = 0

        self.egocam_forward_offset_m = egocam_forward_offset_m
        self.vehicle_color = vehicle_color
        self.meters_to_pixels = meters_to_pixels

        #init renderer:
        self.renderer = openGL3d(track_png_path=track_png_path,
                                    track_size_px=track_size_px,
                                    left_guardrail=None,
                                    right_guardrail=None,
                                    render_size=egocam_render_size,
                                    fov_y_deg=egocam_fov_y_deg, 
                                    fx=egocam_fx,
                                    fy=egocam_fy,
                                    cx=egocam_cx,
                                    cy=egocam_cy,
                                    distortion=egocam_distortion,
                                    skybox_seed=egocam_skybox_seed)

    def _create_egocam_image(self, x:int, y:int, z:int, yaw_rad:float, pitch_rad: float=0.0):
        self.renderer.set_pose( x=x,
                                y=y,
                                z=z,
                                yaw_rad=yaw_rad,
                                pitch_rad=pitch_rad)
        
        bgr = self.renderer.render() # this is the rendered image as seen by the ego-cam
                                     # and thus input to the machine learning algorithm
        return bgr

    def _create_vehicle(self):
        self.v = Vehicle(self.initial_pose_x_y_yaw[0], 
                         self.initial_pose_x_y_yaw[1], 
                         self.initial_pose_x_y_yaw[2], 
                         self.meters_to_pixels, 
                         self.racetrack)
        self.v.color = self.vehicle_color
        self.v.setPhysicalProperties(vehicles[self.v.color]["length_m"], 
                                     vehicles[self.v.color]["width_m"], 
                                     vehicles[self.v.color]["rear_axle_offset_m"], 
                                     vehicles[self.v.color]["steering_measurements"], 
                                     vehicles[self.v.color]["min_motor_value"],
                                     vehicles[self.v.color]["max_motor_value"],
                                     vehicles[self.v.color]["speed_factor"],
                                     vehicles[self.v.color]["min_steering_angle"],
                                     vehicles[self.v.color]["max_steering_angle"],
                                     vehicles[self.v.color]["steering_inverted"],
                                     vehicles[self.v.color]["height_over_ground_colored_marker_m"],
                                     vehicles[self.v.color]["height_over_ground_white_marker_m"],
                                     vehicles[self.v.color]["delay_ms"]
                                     )
        
        #setup motor parameters
        self.v.initMotorPID(vehicles[self.v.color]["motor_pid"][0],
                            vehicles[self.v.color]["motor_pid"][1],
                            vehicles[self.v.color]["motor_pid"][2],
                            error_history_length=50)
        
        #setup steering parameters
        self.v.initSteeringPID(vehicles[self.v.color]["steering_pid"][0],
                               vehicles[self.v.color]["steering_pid"][1],
                               vehicles[self.v.color]["steering_pid"][2],
                               error_history_length=50)
        
        #setup controller
        self.v.controller = DisparityExtender(car_width=(vehicles[self.v.color]["width_m"])*self.meters_to_pixels, 
                                              disparity_threshold=50, 
                                              tolerance=20)
        
        #setup lidar
        self.v.lidar = LidarSensor(track=self.racetrack,
                                   field_of_view_deg=vehicles[self.v.color]["lidar_field_of_view_deg"], 
                                   numRays=vehicles[self.v.color]["lidar_numRays"], 
                                   rayLength_px=vehicles[self.v.color]["lidar_rayLength_m"] * self.meters_to_pixels)


    def reset(self):
        """Reset Simulation for the next run"""
        self._create_vehicle()
        self.current_motor_value = 0
        self.last_progress = 0
        self.current_time = 0

        v : Vehicle = self.v

        #create image
        #y-axis in opengl is flipped - hence need to invert yaw, aswell.
        bgr = self._create_egocam_image( x=v.x,
                                y=self.vertical_resolution_px-v.y,
                                z=int(self.height_over_ground_camera_m*self.meters_to_pixels),
                                yaw_rad=v.yaw *-1,
                                pitch_rad=self.egocam_pitch_rad)


        ret = {
            "progress": 0,
            "on_track_flag" : True,
            "over_time_limit": False,
            "current_time": self.current_time,
            "vehicle_speed_mps": v.vehicle_speed,
            "vehicle_yaw_rate": v.yaw_rate,
            "vehicle_yaw": v.yaw,
            "simulation_over": False,
            "camera_image_bgr": bgr
        }

        return ret
        

    def nextStep(self, target_velocity_mps:float, target_steering_angle_rad:float):
        """Simulate the next timestep"""
        v : Vehicle = self.v  #alias

        ############# STEP 1 #####################
        # visualization: create ego camera image #
        ##########################################

        #y-axis in opengl is flipped - hence need to invert yaw, aswell.

        camera_offset_px = self.egocam_forward_offset_m * self.meters_to_pixels
        camera_x = v.x + math.cos(v.yaw) * camera_offset_px
        camera_y = v.y + math.sin(v.yaw) * camera_offset_px

        bgr = self._create_egocam_image(x=camera_x,
                            y=self.vertical_resolution_px-camera_y,
                            z=int(self.height_over_ground_camera_m*self.meters_to_pixels),
                            yaw_rad=v.yaw *-1,
                            pitch_rad=self.egocam_pitch_rad)    # Pitch on frontcam to track



        ############# STEP 2 ##############
        # apply actions to the vehicle    #
        ###################################

        #TODO: Currently, there is NO delay simulated between sending the commands to the vehicle and applying those commands.
        #       In the real setup, there will be some delay between making sensor measurements, sending those to the
        #       processing computer, computing the actions, sending actions back to the vehicle, and applying the actions
        #       to the servo and motor.


        #apply actions to simulated vehicle
        current_motor_value = v.sendControlsToHardware(target_velocity_mps=target_velocity_mps,
                                                        target_steering_angle_rad=target_steering_angle_rad,
                                                        current_motor_value=self.current_motor_value,
                                                        emergency_brake=False,
                                                        simulate=True)
        #remember applied motor value for next iteration
        self.current_motor_value = current_motor_value



        ############# STEP 3 ###############
        # simulate movement of the vehicle #
        ####################################
        

        # simulate movement by assuming a vehicle speed based on motor_value and
        # by calculating the yaw rate which results from the given steering angle
        # NOTE: this is a very simplified model which assumes instant steering- and motorpower changes


        v.vehicle_speed = current_motor_value / 100.0   # wild guess and highly unrealistic, as here the vehicle has NO mass inertia.
                                                        # TODO: implement a more realistic vehicle model, which takes into account the mass of the vehicle and the inertia of the motor and servo.

        if v.steering_inverted:
            target_steering_angle_rad = -target_steering_angle_rad

        # yaw rate (r) ≈ (vehicle speed (U) / wheelbase (L)) * sin(steering angle (δ))
        v.yaw_rate = (v.vehicle_speed / v.wheel_base_m) * math.sin(target_steering_angle_rad)
        v.yaw = unwind_angle(v.yaw + v.yaw_rate*self.dt)


        #simple linear extrapolation in direction of the vehicle.
        #NOTE: This is a very simple model, where the vehicle will never experience under- or oversteering.

        distance = self.dt * v.vehicle_speed * self.meters_to_pixels
        dx, dy = rotate(distance, 0.0, unwind_angle(v.yaw))
        x = v.x + dx
        y = v.y + dy

        #set values
        v.x = int(x)
        v.y = int(y)

        ############# STEP 4 ###############
        # compute Metrics #
        ####################################

        #check if vehicle is still on track
        
        car = box(x - v.length_px/2, y - v.width_px/2, x + v.length_px/2, y + v.width_px/2)

        car = shapely.affinity.rotate(geom=car, angle=v.yaw, use_radians=True)               # rotate vehicle around its center

        on_track = self.racetrack.polygon.contains(car)

        #check where the vehicle is on track (in percent)
        progress = self.progressCalculator.progress(vehicle_xy=(v.x, v.y),
                                                    prev_progress=self.last_progress,
                                                    search_radius=5,
                                                    max_jump_frac=0.05,
                                                    yaw_weight=5.0,
                                                    yaw=v.yaw)
        
        last_progress = progress['progress_frac']

        self.last_progress = last_progress

        self.current_time += self.dt

        #check if over time limit
        over_time_limit = self.current_time >= self.time_limit_s

        prog = 1.0-last_progress
        if self.current_time < 2.0 and prog > 0.8:  #we have a problem at progress=0.0 as it sometimes wrapps arround to 0.999
            prog = 0.0

        ret = {
            "progress": prog,
            "on_track_flag" : on_track,
            "over_time_limit": over_time_limit,
            "current_time": self.current_time,
            "vehicle_speed_mps": v.vehicle_speed,
            "vehicle_yaw_rate": v.yaw_rate,
            "vehicle_yaw": v.yaw,
            "simulation_over": not on_track or over_time_limit,
            "camera_image_bgr": bgr
        }
        return ret

