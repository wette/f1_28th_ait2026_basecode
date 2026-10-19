#ip addresses of vehicles
vehicles = {
        "yellow": {
            "ip": "10.134.137.45",
            "port": 6446,
            "delay_ms": 120,
            "motor_pid": (6, 0, 0),         #PID parameters to control the motor
            "steering_pid": (0.9, 0, 0),  #PID parameters to control the steering
            "steering_inverted": True,
            "length_m": 0.18,           #vehicle length
            "width_m":  0.08,           #vehicle width
            "rear_axle_offset_m" : 0.065, #rear-center offset of the center of rear axle (where the black Dot on the vehicle is)
            "height_over_ground_white_marker_m": 0.025,
            "height_over_ground_colored_marker_m": 0.042,
            "steering_measurements" : [(180, -24.67991246626858), (130, -10.119637971863794),(90, 5.179684238656864),(50, 25.66474028449667), (10, 41.47479972843025) ],
            "max_steering_angle": 27,  #links
            "min_steering_angle": -27, #rechts
            "min_motor_value" : 60,         #min motor voltage (0-255) to get the vehicle moving
            "max_motor_value" : 110,        #max motor voltage (0-255)
            "speed_factor": 1.0,
            "lidar_field_of_view_deg": 160, 
            "lidar_numRays": 60, 
            "lidar_rayLength_m" : 1.0
        },
        "green": {
            "ip": "10.134.137.41",
            "port": 6446,
            "delay_ms": 150,
            "motor_pid": (6, 0, 0),         #PID parameters to control the motor
            "steering_pid": (1.0, 0.0001, 0),  #PID parameters to control the steering
            "steering_inverted": False,
            "length_m": 0.18,           #vehicle length
            "width_m":  0.08,           #vehicle width
            "rear_axle_offset_m" : 0.065, #rear-center offset of the center of rear axle (where the black Dot on the vehicle is)
            "height_over_ground_white_marker_m": 0.025,
            "height_over_ground_colored_marker_m": 0.042,
            "steering_measurements" :  [(10, -46.0), (18, -46.0), (26, -46.0), (34, -37.0), (42, -31.0), (50, -28.0), (58, -21.0), (66, -16.0), (74, -12.0), (82, -7.0), (90, 0.0), (98, 2.0), (106, 11.0), (114, 13.0), (122, 21.0), (130, 25.0), (138, 35.0), (146, 38.0), (154, 40.0), (162, 40.0), (170, 40.0)],
            "max_steering_angle": 35,
            "min_steering_angle": -35,
            "min_motor_value" :60,         #min motor voltage (0-255) to get the vehicle moving
            "max_motor_value" : 110,        #max motor voltage (0-255)
            "speed_factor": 1.0,
            "lidar_field_of_view_deg": 140, 
            "lidar_numRays": 60, 
            "lidar_rayLength_m" : 1.0
        },
        "light-blue": {
            "ip": "10.134.137.42",
            "port": 6446,
            "delay_ms": 130,
            "motor_pid": (6, 0, 0),         #PID parameters to control the motor
            "steering_pid": (0.9, 0, 0),  #PID parameters to control the steering
            "steering_inverted": True,
            "length_m": 0.18,           #vehicle length
            "width_m":  0.08,           #vehicle width
            "rear_axle_offset_m" : 0.065, #rear-center offset of the center of rear axle (where the black Dot on the vehicle is)
            "height_over_ground_white_marker_m": 0.025,
            "height_over_ground_colored_marker_m": 0.042,
            "steering_measurements" : [(180, -24.638332242378844),(130, -16.28434769293722), (90, 3.4834823178040932), (50, 21.1302691899593), (10, 32.390484557380965)],
            "max_steering_angle": 27,  #links
            "min_steering_angle": -27, #rechts
            "min_motor_value" : 50,         #min motor voltage (0-255) to get the vehicle moving
            "max_motor_value" : 120,        #max motor voltage (0-255)
            "speed_factor": 1.0,
            "lidar_field_of_view_deg": 160, 
            "lidar_numRays": 80, 
            "lidar_rayLength_m" : 1.0
        },
        "blue": {
            "ip": "10.134.137.43",
            "port": 6446,
            "delay_ms": 120,
            "motor_pid": (6, 0, 0),         #PID parameters to control the motor
            "steering_pid": (0.9, 0, 0),  #PID parameters to control the steering
            "steering_inverted": True,
            "length_m": 0.18,           #vehicle length
            "width_m":  0.08,           #vehicle width
            "rear_axle_offset_m" : 0.065, #rear-center offset of the center of rear axle (where the black Dot on the vehicle is)
            "height_over_ground_white_marker_m": 0.025,
            "height_over_ground_colored_marker_m": 0.042,
            "steering_measurements" : [(180, -24.67991246626858), (130, -10.119637971863794),(90, 5.179684238656864),(50, 25.66474028449667), (10, 41.47479972843025) ],
            "max_steering_angle": 27,  #links
            "min_steering_angle": -27, #rechts
            "min_motor_value" : 60,         #min motor voltage (0-255) to get the vehicle moving
            "max_motor_value" : 130,        #max motor voltage (0-255)
            "speed_factor": 1.1,
            "lidar_field_of_view_deg": 160, 
            "lidar_numRays": 60, 
            "lidar_rayLength_m" : 1.0
                },
}
