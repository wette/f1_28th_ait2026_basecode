import math
import cv2 as cv
import numpy as np

def distance(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]

    return math.sqrt(dx**2 + dy**2)

#get yaw from two 2d points (pointing from point1 to point2)
def getyaw(point1, point2):
    a = point2[1] - point1[1]
    b = point2[0] - point1[0]
    yaw = -1*math.atan2(b, a)  + math.pi/2.0
    return yaw

#rotate vector by angle
def rotate(x: float, y: float, alpha:float) -> tuple[float, float]:
    xrot = x*math.cos(alpha) - y*math.sin(alpha)
    yrot = x*math.sin(alpha) + y*math.cos(alpha)
    return xrot, yrot

#h is hue in range 0 to 180
#returns bgr in rage 0 to 255
def hueToBGR(h: float):
    color = cv.cvtColor(np.uint8([[np.array([int(h), 255,  255], dtype=np.uint8)]]), cv.COLOR_HSV2BGR)
    return int(color[0][0][0]), int(color[0][0][1]), int(color[0][0][2])

#shift angle in range [-pi, +pi]
def unwind_angle(alpha_rad : float, start=-math.pi, end=math.pi) -> float:
    while alpha_rad < start:
        alpha_rad += math.pi*2
    while alpha_rad > end:
        alpha_rad -= math.pi*2
    
    return alpha_rad

def get_average_angle(angles_rad: float) -> float:
    x = 0.0
    y = 0.0
    for alpha in angles_rad:
        x += math.cos(alpha)
        y += math.sin(alpha)
    average_angle_rad = -1*math.atan2(x,y)  + math.pi/2.0
    return average_angle_rad


def difference_in_angle(angle1: float, angle2: float) -> float:
    """Computes the shortest signed difference (angle2 - angle1) in radians.
    
    Returns a value in the range [-pi, pi].
    """
    # Using atan2(sin(diff), cos(diff)) avoids branch discontinuities 
    # and handles arbitrary angle wrapping robustly without numerical instability.
    return math.atan2(math.sin(angle2 - angle1), math.cos(angle2 - angle1))


def createPlot(frame, 
               plot_x:int, 
               plot_y:int, 
               plot_height:int, 
               title:str, 
               colors : list[list[int]], 
               values : list[list[float]], 
               dx=3,
               moving_average=1):
    dy = plot_height

    cv.putText(frame, title, (plot_x, plot_y-dy-10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
            colors[0], 1, cv.LINE_AA)

    numMaxValues = len(values[0])

    #draw background(semi transparent):
    # First we crop the sub-rect from the image
    x, y, w, h = plot_x, plot_y-dy, dx*numMaxValues, dy
    sub_img = frame[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv.addWeighted(sub_img, 0.3, white_rect, 0.7, 1.0)

    # Putting the image back to its position
    frame[y:y+h, x:x+w] = res


    #draw x and y axis:
    cv.line(frame, [plot_x, plot_y], [plot_x+dx*numMaxValues, plot_y], (255,255,255), 1) #x-axis
    cv.line(frame, [plot_x, plot_y], [plot_x, plot_y-dy], (255,255,255), 1) #y-axis

    #plot actual data
    for j in range(0, len(values)):
        for i in range(0, len(values[0])-1-moving_average):
            last_val = int(sum(values[j][i : i+moving_average])/moving_average)
            act_val = int(sum(values[j][i+1 : i+1+moving_average])/moving_average)
            cv.line(frame, 
                    [plot_x + i*dx, plot_y - last_val], 
                    [plot_x + (i+1)*dx, plot_y - act_val], 
                    colors[j], 
                    1)

    return frame


def hue_mask_hsv(hsv_img, hue_value, tol=10):
    """
    Erstellt eine Binärmaske (uint8, 0/255) für einen Ziel-Hue-Wert +/- tol
    auf einem HSV-Bild (OpenCV-Format: H in [0..179]) inkl. Wrap-around.

    Parameter:
      hsv_img   : HSV-Bild (np.ndarray, dtype=uint8), z.B. aus cv2.cvtColor(BGR->HSV)
      hue_value : Ziel-Hue (int). Üblich: 0..179 (OpenCV-Skala). Werte außerhalb werden gemoddet.
      tol       : Toleranz (int), z.B. 10

    Rückgabe:
      mask : Binärmaske (np.ndarray, dtype=uint8) mit 0 oder 255
    """
    if hsv_img is None or hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("hsv_img muss ein HSV-Farbbild mit 3 Kanälen sein.")

    # OpenCV-Hue ist zyklisch in [0..179]
    HMAX = 180
    h = int(hue_value) % HMAX
    t = int(tol)

    # Grenzen berechnen
    lower = h - t
    upper = h + t

    # optional: S/V nicht einschränken (0..255)
    low_sv = (0, 0)
    high_sv = (255, 255)

    if 0 <= lower and upper < HMAX:
        # Kein Wrap-around
        lowerb = (lower, *low_sv)
        upperb = (upper, *high_sv)
        mask = cv.inRange(hsv_img, np.array(lowerb, dtype=np.uint8),
                                  np.array(upperb, dtype=np.uint8))
    else:
        # Wrap-around: zwei Bereiche
        # Beispiel: h=5,t=10 => [-5..15] -> [0..15] U [175..179]
        lower1 = (max(lower, 0), *low_sv)
        upper1 = (min(upper, HMAX - 1), *high_sv)

        lower2 = ((lower + HMAX) % HMAX, *low_sv)  # in den oberen Bereich verschoben
        upper2 = (HMAX - 1, *high_sv)

        # Falls upper >= 180 (z.B. h=175,t=10 => [165..185] -> [165..179] U [0..5])
        if upper >= HMAX:
            lower1 = (lower, *low_sv)
            upper1 = (HMAX - 1, *high_sv)

            lower2 = (0, *low_sv)
            upper2 = (upper % HMAX, *high_sv)

            mask1 = cv.inRange(hsv_img, np.array(lower1, dtype=np.uint8),
                                        np.array(upper1, dtype=np.uint8))
            mask2 = cv.inRange(hsv_img, np.array(lower2, dtype=np.uint8),
                                        np.array(upper2, dtype=np.uint8))
            mask = cv.bitwise_or(mask1, mask2)
        else:
            mask1 = cv.inRange(hsv_img, np.array(lower1, dtype=np.uint8),
                                        np.array(upper1, dtype=np.uint8))
            mask2 = cv.inRange(hsv_img, np.array(lower2, dtype=np.uint8),
                                        np.array(upper2, dtype=np.uint8))
            mask = cv.bitwise_or(mask1, mask2)

    return mask
