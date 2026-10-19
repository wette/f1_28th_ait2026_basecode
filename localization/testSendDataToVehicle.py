import socket
import time


UDP_IP = "10.134.137.42"
UDP_PORT = 6446

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

current_motor_value = 0
steering_angle = 40

output = [0, current_motor_value, steering_angle]
msg = ";".join(f"{e:03}" for e in output)



while True:
    #sweep the steering angle from 30 to 160 pwm:
    for steering_angle in range(30, 161, 1):
        output = [0, current_motor_value, steering_angle]
        msg = ";".join(f"{e:03}" for e in output)
        sock.sendto(bytes(msg, "utf-8"), (UDP_IP, UDP_PORT))
        print(f"Sent: {msg}")
        time.sleep(0.05)
    #sock.sendto(bytes(msg, "utf-8"), (UDP_IP, UDP_PORT))