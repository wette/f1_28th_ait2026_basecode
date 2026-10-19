import cv2 as cv
from control.track import Track
from localization.vehicle import Vehicle
import copy
import time
from localization.camera import Camera

class OverHeadView:
    def __init__(self, backgroundImageFilename:str, track:Track):
        self.frame = cv.imread(backgroundImageFilename)

        #draw racetrack on frame:
        for idx in range(-1, len(track.innerBounds)-1):
            cv.line(self.frame, track.innerBounds[idx], track.innerBounds[idx+1], (255,255,255), 2)

        for idx in range(-1, len(track.outerBounds)-1):
            cv.line(self.frame, track.outerBounds[idx], track.outerBounds[idx+1], (255,255,255), 2)

    def getImage(self, vehicle : Vehicle):
        frame = copy.copy(self.frame)

        boundingbox = vehicle.getBoundingBox()
        color = (0, 255, 0)
        Camera.drawBoundingBox(frame, boundingbox, color=color)
        cv.putText(frame, f"Speed: {vehicle.getSpeed():.2f} m/s", (int(boundingbox[0][0]), int(boundingbox[0][1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 1, cv.LINE_AA)

        return frame
