import math
import numpy
import socket
import struct
import time
import shapely
import copy

from localization.helper_functions import *
from .pid_controller import PIDController

from control.disparity_extender import DisparityExtender
from control.lidarSensor import LidarSensor
from control.track import Track
from control.overtaking_controller import OvertakingController

class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.dx = 0
        self.dy = 0

    def update(self, x, y, radius):
        self.dx = int(x - self.x)
        self.dy = int(y - self.y)
        self.x = int(x)
        self.y = int(y)
        self.radius = int(radius)

    def predict_next_position(self):
        predicted_x = self.x + self.dx
        predicted_y = self.y + self.dy

        return predicted_x, predicted_y


class Vehicle:
    def __init__(self, x, y, yaw, meters_to_pixels, racetrack, front_marker:Circle|None = None, rear_marker:Circle|None = None):
        #position and orientation of the vehicle
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw_rate = 0.0

        #position and orientation of the circles (markers) on the vehicle
        self.front_marker:Circle|None = front_marker
        self.rear_marker:Circle|None = rear_marker

        self.track = racetrack
        
        #IP address
        self.port   = None
        self.IP     = None
        self.sock   = None

        #PID for speed control
        self.motor_pid = None
        self.target_velocity_mps = 0

        #PID for steering
        self.steering_pid = None

        #steering map (steering angle -> PWM)
        self.steering_map : list = None
        self.steering_map_start_angle_rad = None
        self.steering_map_end_angle_rad = None
        self.min_steering_angle_deg = -45
        self.max_steering_angle_deg = 45
        self.steering_inverted = False

        #for filtering steering control
        self.target_steering_angle_rad = 0.0

        #reference to contoroller
        self.controller : None | DisparityExtender = None

        #reference to lidar sensor
        self.lidar : None | LidarSensor = None

        #color of the vehicle
        self.color : None | str  = None

        #convert meters to pixels
        self.meters_to_pixels = meters_to_pixels #how many pixels are in one meter?

        #vehicle dimensions in pixels
        self.length_px = 0.18 * self.meters_to_pixels
        self.width_px = 0.08 * self.meters_to_pixels
        
        self.wheel_base_m = 0.094 #TODO: externalize!

        self.height_over_ground_colored_marker_m = 0
        self.height_over_ground_white_marker_m = 0

        #rear-center offset of the center of rear axle (where the colored Dot on the vehicle is)
        self.rear_axle_offset_px = 0.065 * self.meters_to_pixels

        #current speed of the vehicle
        self.vehicle_speed = 0 #meters per second; absolute speed

        #min and max voltage (0-255) for motor:
        self.min_motor_value = 0
        self.max_motor_value = 0
        self.speedFactor = 1.0  #linear factor to increse/decrease overall target velocity

        self.lateral_acceleration_mps = 0
        self.lateral_speed_mps = 0
        self.longitudinal_acceleration_mps = 0
        self.longitudinal_speed_mps = 0

        self.speed_filter_values = [] #history of speed values to filter
        self.yaw_filter_values = [] #history of yaw values to filter
        self.yaw_rate_filter_values = [] #history of yaw rate values to filter

        self.command_history = [] #history of the issued driving commands
        self.command_frequency_hz = 40.0 #TODO: externalize!

        #data for overtaking
        self.overtaking_follower_counter = 0
        self.overtaking_followee_counter = 0

        #timekeeping
        self.is_on_finish_line = False
        self.time_start_of_lap = time.time()
        self.last_laptime = 0

        self.last_update = 0

        self.ttl = 5

    #gets measurements as done by vehicle_calibration.py and converts into steering map
    def initSteeringMap(self, measurements : list[float]):
        if len(measurements) == 0:
            return
        
        self.steering_map = []
        min_angle = measurements[0][1]
        max_angle = measurements[-1][1]

        self.steering_map_start_angle_rad   = math.radians(min_angle)
        self.steering_map_end_angle_rad     = math.radians(max_angle)

        num_map_entries = 255
        for i in range(0, num_map_entries):
            target_angle = min_angle + ((max_angle-min_angle)/num_map_entries) * i

            #find steering command that leads to target angle
            next_lower_idx = len(measurements)-2
            next_higher_idx = len(measurements)-1

            for j in range(0, len(measurements)-1):
                if measurements[j][1] <= target_angle <= measurements[j+1][1]:
                    #found
                    next_lower_idx = j
                    next_higher_idx = j+1
                    break
            
            #linear interpolation
            delta_deg = measurements[next_higher_idx][1] - measurements[next_lower_idx][1]
            if delta_deg != 0.0:
                diff_deg = target_angle - measurements[next_lower_idx][1]

                delta_contr = measurements[next_higher_idx][0] - measurements[next_lower_idx][0]

                interpolated = measurements[next_lower_idx][0] + (diff_deg/delta_deg) * delta_contr

                self.steering_map.append(interpolated)
            else:
                #same values - interpolation is the actual value itself.
                self.steering_map.append(measurements[next_lower_idx][0])

    #convert steering angle to control value
    def getSteeringControlFromMap(self, target_angle_rad):
        num_map_entries = 255

        index = int(((target_angle_rad - self.steering_map_start_angle_rad) / (self.steering_map_end_angle_rad - self.steering_map_start_angle_rad)) * num_map_entries)
        index = max(0, min(index, num_map_entries-1))

        return self.steering_map[index]




    def compute_next_command(self, delta_t : float, opponents : list['Vehicle'] = []):
        x, y, yaw = self.getPositionEstimateCommandHistory(self.last_update + delta_t, longitudinal_offset_m=0.03) #TODO: find out which delta is required here!

        #do not include opponents, if we are currently overtaken
        op = opponents
        if self.overtaking_followee_counter > 0:
            op = []
        distances, rays = self.lidar.getReadings(x, y, yaw, opponents = op)

        #ask controller what to do
        setpoint, target_steering_angle_rad = self.controller.compute_steering(copy.copy(distances), rays)

        if setpoint is None:
            return 0.0, 0.0, None, None, False

        target_steering_angle_rad = target_steering_angle_rad - self.yaw

        #filter target angle:
        #new_angle = old_angle*0.5+new_angle*0.5
        target_steering_angle_rad = get_average_angle([target_steering_angle_rad, self.target_steering_angle_rad])

        #unwind target angle
        target_steering_angle_rad = unwind_angle(target_steering_angle_rad)


        #compute target velocity
        target_velocity_mps = 0.8

        #find out how fast we can go: TODO: find out more reasonable numbers.
        target_steering_angle_deg = abs(math.degrees(target_steering_angle_rad))
        if 0 <= target_steering_angle_deg < 2:
            target_velocity_mps = 1.6
        if 2 <= target_steering_angle_deg < 5:
            target_velocity_mps = 1.4
        if 5 <= target_steering_angle_deg < 10:
            target_velocity_mps = 1.2
        if 10 <= target_steering_angle_deg < 20:
            target_velocity_mps = 1.0
        if 30 <= target_steering_angle_deg:
            target_velocity_mps = 0.8


        old_target_steering_angle_rad = target_steering_angle_rad

        #apply lookahead to smooth out steering commands
        
        lookahead_m = target_velocity_mps * 0.20
        setpoint_distance = math.sqrt( (self.x-setpoint[0])**2 + (self.y-setpoint[1])**2 ) / self.meters_to_pixels
        #print(lookahead_m, setpoint_distance)
        lookahead_m = min(lookahead_m, max(0.1, setpoint_distance))
        steering_angle = math.atan2(2.0 * self.wheel_base_m * math.sin(target_steering_angle_rad), lookahead_m) #pure pursuit formula
        target_steering_angle_rad = steering_angle
        


        #apply speedfactor
        target_velocity_mps *= self.speedFactor


        #be aware of opponents on the track:
        opponent_we_are_following = self.isFollowingAnOpponent(opponents)
        followed_by_opponent = self.isFollowedByAnOpponent(opponents)

        OVERTAKE_TIME = 100
        

        #if we are close to an opponent - just follow
        if opponent_we_are_following is not None:
            #target_velocity_mps = 0.2
            
            if self.overtaking_followee_counter == 0: #otherwise we have just been passed.
                print(f"{self.color}: following")
                self.overtaking_follower_counter = OVERTAKE_TIME
        else:
            self.overtaking_follower_counter = max(self.overtaking_follower_counter-1, 0)
        
        if followed_by_opponent is not None:
            if self.overtaking_follower_counter == 0:
                print(f"{self.color}: i have a follower")
                self.overtaking_followee_counter = OVERTAKE_TIME
        else:
            self.overtaking_followee_counter = max(self.overtaking_followee_counter-1, 0)


        #test new overtaking controller:
        # if we are between 10 and 20 cm behind an opponent, activate overtaking controller.
        #the controller will get the vehicle in a position where it can overtake using the normal controller
        """
        max_target_speed_mps = None
        if opponent_we_are_following and self.overtaking_followee_counter == 0:
            #activate overtake controller
            overtake_ctrl = OvertakingController(self.width_px/self.meters_to_pixels,
                                                        self.track,
                                                        self.meters_to_pixels)
            setpoint, target_steering_angle_rad, max_target_speed_mps = overtake_ctrl.compute_steering(self.x, self.y, self.yaw, opponent_we_are_following)
                              

        #to ease overtaking: a vehicle which is getting followed will slow down to let the other pass
        if self.overtaking_followee_counter > 0:
            target_velocity_mps = max(0.5, target_velocity_mps - (self.overtaking_followee_counter/OVERTAKE_TIME) * 0.4)

        if max_target_speed_mps is not None:
            target_velocity_mps = min(target_velocity_mps, max_target_speed_mps)
        """

        #check for emeregency brake: if the rays right in front of the vehicle (+/- 8 deg) are close to zero
        emeregency_brake = False

        
        required_space_px = 0.005*self.meters_to_pixels #don't be too much on the grass

        if opponent_we_are_following is not None:
            #we are closely following an oppponent - keep a ceertain distance
            required_space_px = 0.04*self.meters_to_pixels


        total_num_rays = len(distances)
        lidar_fov_rad = self.lidar.fov_rad
        relevant_num_rays = int( (math.radians(16)/lidar_fov_rad) * total_num_rays )
        start_idx = max(int(total_num_rays/2) - int(relevant_num_rays/2), 0)
        end_idx = min(int(total_num_rays/2) + int(relevant_num_rays/2), total_num_rays)
        available_space_px = np.min(distances[start_idx:end_idx])
        #print(total_num_rays, relevant_num_rays, start_idx, end_idx, available_space_px, required_space_px)
        #print(distances[start_idx:end_idx])
        if available_space_px < required_space_px:
            #brake!
            if self.getSpeed() > 0.1:  #make sure we never come to a true full stop.
                emeregency_brake = True
                print("emer brake", total_num_rays, relevant_num_rays, start_idx, end_idx, available_space_px, required_space_px)



        self.target_velocity_mps = target_velocity_mps
        self.target_steering_angle_rad = target_steering_angle_rad

        return target_velocity_mps, target_steering_angle_rad, rays, setpoint, emeregency_brake


    def isFollowingAnOpponent(self, opponents : list['Vehicle']) -> 'Vehicle':
        closestOpponent = None
        for opp in opponents:
            dist = distance([self.x, self.y], [opp.x, opp.y])/self.meters_to_pixels
            angle = getyaw([self.x, self.y], [opp.x, opp.y])
            angle = abs(difference_in_angle(angle, self.yaw))

            v_len = self.length_px/self.meters_to_pixels/2.0
            if angle < math.radians(90) and (0.1+v_len <= dist <= 0.2+v_len):
                closestOpponent = opp
                break

        return closestOpponent
    
    def isFollowedByAnOpponent(self, opponents : list['Vehicle']) -> 'Vehicle':
        for opp in opponents:
            if opp.isFollowingAnOpponent([self]):
                 return opp

        return None


    def setPhysicalProperties(self, length_m, 
                                    width_m, 
                                    rear_axle_offset_m, 
                                    steering_measurements, 
                                    min_motor_value,
                                    max_motor_value, 
                                    speedFactor,
                                    min_steering_angle_deg,
                                    max_steering_angle_deg,
                                    steering_inverted,
                                    height_over_ground_colored_marker_m,
                                    height_over_ground_white_marker_m,
                                    delay_ms):
        self.length_px = length_m * self.meters_to_pixels
        self.width_px = width_m * self.meters_to_pixels
        self.rear_axle_offset_px = rear_axle_offset_m * self.meters_to_pixels
        self.min_motor_value = min_motor_value
        self.max_motor_value = max_motor_value
        self.speedFactor = speedFactor
        self.min_steering_angle_deg = min_steering_angle_deg
        self.max_steering_angle_deg = max_steering_angle_deg
        self.steering_inverted = steering_inverted
        self.height_over_ground_colored_marker_m = height_over_ground_colored_marker_m
        self.height_over_ground_white_marker_m = height_over_ground_white_marker_m
        self.delay_ms = delay_ms

        self.initSteeringMap(steering_measurements)

    def getPosition(self):
        return self.x, self.y
    
    def getOrientation(self):
        return self.yaw
    
    def getSpeed(self):
        return self.vehicle_speed
    
    def updateMarkers(self, front_x, front_y, front_radius, rear_x, rear_y, rear_radius):
        if self.front_marker is not None and self.rear_marker is not None:
            self.front_marker.update(front_x, front_y, front_radius)
            self.rear_marker.update(rear_x, rear_y, rear_radius)
        else:
            self.front_marker = Circle(front_x, front_y, front_radius)
            self.rear_marker = Circle(rear_x, rear_y, rear_radius)
            
    def removeMarkers(self):
        self.front_marker = None
        self.rear_marker = None
    
    def updatePose(self, x, y, yaw, current_time):
        dx = x - self.x
        dy = y - self.y
        dt = current_time - self.last_update
        dist = math.sqrt(dx**2 + dy**2)
        if dt > 0:
            speed = dist / dt
        else:
            speed = self.vehicle_speed

        if speed < 0.05:
            speed = 0.0     #standstill

        speed = speed / self.meters_to_pixels #convert to m/s

        #compute lateral speed and lateral acceleration
        #current position in coordinate frame of last position to find out lateral and longitudinal acc
        if dt != 0:
            mx, my = rotate(x - self.x, y - self.y, -1 * self.yaw)
            mx /= self.meters_to_pixels #convert to m/s
            my /= self.meters_to_pixels #convert to m/s
            self.lateral_acceleration_mps = my / dt - self.lateral_speed_mps
            self.lateral_speed_mps = my / dt
            self.longitudinal_acceleration_mps = mx / dt - self.longitudinal_speed_mps
            self.longitudinal_speed_mps = mx / dt

        self.speed_filter_values.append(speed)
        self.yaw_filter_values.append(yaw)
        self.speed_filter_values = self.speed_filter_values[-15:] #keep 5 most recent values
        self.yaw_filter_values = self.yaw_filter_values[-5:] #keep 5 most recent values

        self.x = x
        self.y = y
        self.vehicle_speed = sum(self.speed_filter_values) / len(self.speed_filter_values) #average
        old_yaw = self.yaw
        self.yaw = unwind_angle(get_average_angle(self.yaw_filter_values)) #average
        self.yaw_rate = self.yaw_rate if dt == 0.0 else (self.yaw - old_yaw)/dt
        self.last_update = current_time

    
    #returns projected x,y,yaw for time at_time (center of the rear axle)
    #longitudinal_offset_m adds a distance in front of the vehicle (where vehile is the center of the rear axle)
    def getPositionEstimate(self, at_time, longitudinal_offset_m=0.0):
        #TODO: implement constant velocity and turn rate model
        #for now simple linear estimation...
        dt = at_time - self.last_update
        x1, y1, yaw1 = None, None, None

        #simple linear extrapolation in direction of the vehicle.
        distance = (dt * self.vehicle_speed + longitudinal_offset_m) * self.meters_to_pixels
        yaw_rate = (self.vehicle_speed / self.wheel_base_m) * math.sin(self.target_steering_angle_rad)
        dx, dy = rotate(distance, 0.0, unwind_angle(self.yaw + yaw_rate*dt))
        x1 = self.x + dx
        y1 = self.y + dy
        yaw1 = unwind_angle(self.yaw + yaw_rate*dt)

        return x1, y1, yaw1
    
#returns projected x,y,yaw for time at_time (center of the rear axle)
    #dt_commands = time between two sent driving commands
    def getPositionEstimateCommandHistory(self, at_time, longitudinal_offset_m=0.0):
        dt_commands = 1.0/self.command_frequency_hz

        dt = at_time - self.last_update
        x1, y1, yaw1 = self.x, self.y, self.yaw

        n = int(dt/dt_commands)

        if n >= len(self.command_history):
            return self.getPositionEstimate(at_time, longitudinal_offset_m)
        
        #add longitudinal offset:
        distance = longitudinal_offset_m * self.meters_to_pixels
        dx, dy = rotate(distance, 0.0, self.yaw)

        #do not add longitudinal offset outside of track boundaries.
        if not self.lidar.track.pointInsideTrack(shapely.Point(x1+dx,y1+dy)):
            x1 += dx
            y1 += dy

        #simple linear extrapolation in direction of the vehicle over the history of driving commands.
        for i in range(len(self.command_history)-n, len(self.command_history)):
            speed = self.command_history[i][0]
            steering_angle = self.command_history[i][1] * 0.8  #TODO: externalize constant!
            yaw = steering_angle + self.yaw #absolute yaw
            distance = (dt_commands * speed) * self.meters_to_pixels
            dx, dy = rotate(distance, 0.0, unwind_angle(yaw))

            #only extrapolate inside of track boundaries.
            if not self.lidar.track.pointInsideTrack(shapely.Point(x1+dx,y1+dy)):
                break

            x1 += dx
            y1 += dy

            yaw_rate = speed/self.wheel_base_m * math.tan(steering_angle)
            yaw1 += yaw_rate*dt_commands

        #if, for some reason we are stimm outside of the track, set the vehicle back on track.
        while not self.lidar.track.pointInsideTrack(shapely.Point(x1+dx,y1+dy)):
            distance = 0.01 * self.meters_to_pixels #set one cm back.
            dx, dy = rotate(distance, 0.0, self.yaw)
            x1 -= dx
            y1 -= dy


        return x1, y1, unwind_angle(yaw1)
    
    #returns coordinates of a boundingbox of the vehicle.
    #at at_time is not None: projected into future.
    def getBoundingBox(self, at_time = None, use_command_history=True):
        pos_x, pos_y = self.getPosition()
        yaw = self.getOrientation()
        if at_time is not None:
            if use_command_history:
                pos_x, pos_y, yaw = self.getPositionEstimateCommandHistory(at_time)
            else:
                pos_x, pos_y, yaw = self.getPositionEstimate(at_time)

        #define boundingbox in vehicle coordinate system (vehicle drives along x axis)
        bottom_left  = numpy.array([ -self.rear_axle_offset_px, int(-self.width_px/2) ])
        bottom_right = numpy.array([ -self.rear_axle_offset_px,  int(self.width_px/2) ])
        top_left     = numpy.array([ -self.rear_axle_offset_px+self.length_px, int(-self.width_px/2)])
        top_right    = numpy.array([ -self.rear_axle_offset_px+self.length_px,  int(self.width_px/2)])

        #rotate coordinates by yaw angle
        rotMatrix = numpy.array([[numpy.cos(yaw), -numpy.sin(yaw)], 
                         [numpy.sin(yaw),  numpy.cos(yaw)]])
        
        bottom_left = numpy.dot(rotMatrix, bottom_left)
        bottom_right = numpy.dot(rotMatrix, bottom_right)
        top_left = numpy.dot(rotMatrix, top_left)
        top_right = numpy.dot(rotMatrix, top_right)

        #move to x,y position
        bbox = [ list(bottom_left), list(bottom_right), list(top_right), list(top_left) ]
        for p in bbox:
            p[0] += pos_x
            p[1] += pos_y

        return bbox
    
    #initialize the PID controler used to control the speed of the vehicle
    def initMotorPID(self, kp:float, ki:float, kd:float, error_history_length:int):
        self.motor_pid = PIDController(kp, ki, kd, error_history_length)

    #initialize the PID controler used to control the angle of the vehicle
    def initSteeringPID(self, kp:float, ki:float, kd:float, error_history_length:int):
        self.steering_pid = PIDController(kp, ki, kd, error_history_length)

    def initNetworkConnection(self, ip_adress:str, port:int):
        self.port = port
        self.IP = ip_adress
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP

    # send controls via UDP to the vehicle hardware.
    # target_velocity_mps:          target velocity of the vehicle. in meters/second
    # target_steering_angle_rad:    target steering angle of the vehicle. in map coordiates.
    #                               positive is to the right
    # current_motor_value:          the (voltage) value last set for the vehicle
    # emergency_brake:              true to bring vehicle to complete stop.
    #
    # returns:
    # current_motor_value:          the value just set for the motor
    def sendControlsToHardware(self, target_velocity_mps:float, target_steering_angle_rad:float, current_motor_value:int, emergency_brake:bool = False, simulate=False):

        if emergency_brake:
            target_velocity_mps = 0.0

        #ask PID control what to do with the motor voltage
        delta_motor_value = self.motor_pid.update(target_velocity_mps - self.vehicle_speed)
        current_motor_value += delta_motor_value

        #clip - keep a minimum motor value to prevent the vehicle from getting stuck.
        #but only if the target_velocity_mps is greater zero. otherwise we would prevent hard braking
        if target_velocity_mps >= 0.01:
            current_motor_value = int(max(self.min_motor_value, min(self.max_motor_value, current_motor_value))) 
        else:
            current_motor_value = int(max(0, min(self.max_motor_value, current_motor_value))) 
        

        if emergency_brake:
            current_motor_value = 0.0

        #print(f"unwound angle: {math.degrees(target_steering_angle_rad)} - ", end="")

        #filtering the steering commands.
        target_steering_angle_rad = self.steering_pid.update(target_steering_angle_rad)        

        #check steering angle is in bounds
        target_steering_angle_rad = max(self.steering_map_start_angle_rad, min(self.steering_map_end_angle_rad, target_steering_angle_rad))
        #print(f"bounds angle: {math.degrees(target_steering_angle_rad)} - ", end="")
        

        #convert steering angle to number between 10 and 170 (for some reason...), 90 beeing 0°, 10 beeing max left, 170 max right
        #target_steering_angle_rad can be between -self.max_steering_angle_rad and +self.max_steering_angle_rad
        steering_angle = self.getSteeringControlFromMap(target_steering_angle_rad)

        steering_angle = min(steering_angle, 170) #clip at 170
        steering_angle = max(steering_angle, 10)  #clip at 10

        

        #print(f"PWM steering: {steering_angle}, PWM Motor: {current_motor_value}")

        #steering_angle = int(((target_steering_angle_rad/self.max_steering_angle_rad + 1.0) / 0.5) * 160)+10 #TODO: this is most likely NOT what the vehicle expects!

        #send data over to hardware
        
        #binary protocol
        #msg = bytes()
        #msg += struct.pack("!H", self.current_motor_value)    #network-byte-order unsigned 16bit int
        #msg += struct.pack("!H", steering_angle)              #network-byte-order unsigned 16bit int

        #somehow the students decided to use an utf-8 format?!... TODO: find out what the first value (called "status" in their implementation) actually means and convert to binary format.
        output = [0, current_motor_value, steering_angle]
        msg = ";".join(f"{e:03}" for e in output)
        
        if not simulate:
            self.sock.sendto(bytes(msg, "utf-8"), (self.IP, self.port))

        estimated_speed = target_velocity_mps*0.6+self.getSpeed()*0.4 #somewhere between current and target velocity...
        self.command_history.append( (estimated_speed, target_steering_angle_rad) )

        return current_motor_value
