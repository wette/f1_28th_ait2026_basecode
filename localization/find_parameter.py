import cv2 as cv
import numpy as np
import time

path = "/Users/wette/Documents/FHBielefeld/eigeneVorlesungen/AutonomeFahrzeuge1zu32/controlSoftware/localization/webcam.png"
path = '/Users/wette/Desktop/Bildschirmfoto 2025-06-03 um 14.43.38.png'
original = cv.imread(path)

#read first frame from video:
cap = cv.VideoCapture('/Users/wette/Documents/FHBielefeld/eigeneVorlesungen/AutonomeFahrzeuge1zu32/loose_markers.avi')
ret, original = cap.read()

#make image smaller
#original = cv.resize(original, None, fx= 0.3, fy= 0.3, interpolation= cv.INTER_LINEAR)

#parameters
#works: parameter_dp=1.5, minDist=2, param1=300, param2=0.5999999999999999, circle_diameter_px=20, parameter_blur_kernel_size=5 
parameter_dp = 1.5
parameter_minDist = 20
parameter_1 = 300
parameter_2 = 0.8
parameter_blur_kernel_size = 5
circle_diameter_px = 15


def changeParameter(value, keyPressed, keyDown, keyUp, downValue, upValue, minValue, maxValue=100000):
    if keyPressed == ord(keyDown):
        if value - downValue >= minValue:
            return value-downValue
    if keyPressed == ord(keyUp):
        if value + upValue <= maxValue:
            return value + upValue

    return value

while True:
    frame = original.copy()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, parameter_blur_kernel_size)
    #gray = cv.medianBlur(gray, parameter_blur_kernel_size) #2nd blur


    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, parameter_dp, minDist=parameter_minDist, param1=parameter_1, param2=parameter_2, minRadius=int((circle_diameter_px/2.0)*0.5), maxRadius=int((circle_diameter_px/2.0)*1.5))
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            
            if x < frame.shape[1] and y < frame.shape[0]:
                color = frame[y,x]

                cv.circle(gray, (x, y), r, (int(color[0]), int(color[1]), int(color[2])), 2)
                cv.rectangle(gray, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

        
    cv.imshow('frame', gray)
    time.sleep(1/30.0)

    keyPressed = cv.waitKey(1)

    if keyPressed == ord('q'):
        quit()
    
    parameter_dp = changeParameter(parameter_dp, keyPressed,                                'w', 'e', 0.05, 0.05, 0.05)
    parameter_minDist = changeParameter(parameter_minDist, keyPressed,                      's', 'd', 1, 1, 1)
    parameter_1 = changeParameter(parameter_1, keyPressed,                                  'x', 'c', 5, 5, 5)
    parameter_2 = changeParameter(parameter_2, keyPressed,                                  'r', 't', 0.05, 0.05, 0.05, maxValue=1.0)
    circle_diameter_px = changeParameter(circle_diameter_px, keyPressed,                    'f', 'g', 1, 1, 1)
    parameter_blur_kernel_size = changeParameter(parameter_blur_kernel_size, keyPressed,    'v', 'b', 2, 2, 1)



    print(f"parameter_dp={parameter_dp}, minDist={parameter_minDist}, param1={parameter_1}, param2={parameter_2}, circle_diameter_px={circle_diameter_px}, parameter_blur_kernel_size={parameter_blur_kernel_size}       ", end="\r")
        