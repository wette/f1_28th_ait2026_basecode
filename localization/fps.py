import cv2 as cv
import time


cap = cv.VideoCapture(0, cv.CAP_V4L)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv.CAP_PROP_FPS, 90)

t = time.time()
fps = 0.0
while True:
    ret, frame = cap.read()


    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

    dt = time.time()-t

    fps = 0.99*fps + 0.01*(1.0/dt)

    print(f"{fps} FPS at {frame.shape}")
    t = time.time()

cap.close()
