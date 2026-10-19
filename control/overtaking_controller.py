#from sensor import Sensor
import numpy as np
import shapely
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from control.lidarSensor import LidarSensor
from localization.helper_functions import *


class OvertakingController:
    def __init__(self, car_width, track, meters_to_pixels):
        self.car_width = car_width
        self.meters_to_pixels = meters_to_pixels

        self.lidar = LidarSensor(track, 60, 20, 0.3*meters_to_pixels) #TODO: magic numbers.

    #find out if save to pass the oponent on the left or right. if neither, just follow.
    def compute_steering(self, x, y, yaw, opponent : 'Vehicle'):

        #use lidar at each corner of the boundingbox to find out minimum distance to wall
        #(a little overkill, would be more efficient to only cast 2 rays per corner, and compute the distance from that)
        bbox = opponent.getBoundingBox() #bbox = [ list(bottom_left), list(bottom_right), list(top_right), list(top_left) ]
        
        distances_top_left, _  = self.lidar.getReadings(bbox[3][0], bbox[3][1], unwind_angle(opponent.yaw + math.radians(-90)))
        distances_top_right, _ = self.lidar.getReadings(bbox[2][0], bbox[2][1], unwind_angle(opponent.yaw + math.radians(90)))

        distances_bot_left, _  = self.lidar.getReadings(bbox[0][0], bbox[0][1], unwind_angle(opponent.yaw + math.radians(-90)))
        distances_bot_right, _ = self.lidar.getReadings(bbox[1][0], bbox[1][1], unwind_angle(opponent.yaw + math.radians(90)))
        
        distance_left_m  = min(distances_top_left  + distances_bot_left) / self.meters_to_pixels
        distance_right_m = min(distances_top_right + distances_bot_right) / self.meters_to_pixels

        max_target_speed_mps = None

        safety_gap = 0.02

        if distance_left_m < self.car_width+safety_gap and distance_right_m < self.car_width+safety_gap:
            #not possible to drive around the opponent: just follow and wait for a chance to overtake
            setpoint = [opponent.x, opponent.y]
            max_target_speed_mps = opponent.getSpeed()*0.5
            target_steering_angle_rad = unwind_angle(getyaw([x,y], setpoint) - yaw)
            print("#######following#########################")
            return setpoint, target_steering_angle_rad, max_target_speed_mps

        if distance_left_m >= distance_right_m:
            #overtake on the left
            distance = distance_left_m/2.0 * self.meters_to_pixels
            dx, dy = rotate(distance, 0.0, unwind_angle(opponent.yaw + math.radians(-90)))

            setpoint = [ bbox[0][0] + dx, bbox[0][1] + dy] #bot. left

            target_steering_angle_rad = unwind_angle(getyaw([x,y], setpoint) - yaw)
            print("#######left#########################")
            return setpoint, target_steering_angle_rad, max_target_speed_mps

        if distance_right_m > distance_left_m:
            #overtake on the right

            distance = distance_right_m/2.0 * self.meters_to_pixels
            dx, dy = rotate(distance, 0.0, unwind_angle(opponent.yaw + math.radians(90)))

            setpoint = [ bbox[1][0] + dx, bbox[1][1] + dy] #bot. right

            target_steering_angle_rad = unwind_angle(getyaw([x,y], setpoint) - yaw)
            print("#######right#########################")
            return setpoint, target_steering_angle_rad, max_target_speed_mps


        return None, None, None #something went wrong - we should never end up here.