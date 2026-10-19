
import cv2 as cv
import time

NUM_IMAGES = 30
DIR = "/Users/wette/Documents/FHBielefeld/eigeneVorlesungen/AutonomeFahrzeuge1zu32/localization/camera_calibration/images"


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


current_time = time.time()
timer = time.time() + 1.0  # +1 second

i = 0
while i < NUM_IMAGES:
    ret, frame = cap.read()
    current_time = time.time()

    if current_time > timer:
        cv.imwrite(f"{DIR}/image{i}.png", frame)
        i += 1
        timer = time.time() + 1.0  # +1 second
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

