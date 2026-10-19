#from sensor import Sensor
import numpy as np
import shapely
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class DisparityExtender:
    def __init__(self, car_width, disparity_threshold, tolerance):
        self.car_width = car_width
        self.disparity_threshold = disparity_threshold
        self.tolerance = tolerance

    def compute_steering(self, distances, rays):
        disparities_indexes, directions = self.find_disparities(distances)
        modified_distances, setpoint, target_steering_angle_rad = self.extend_disparities(distances, 
                                                                                          rays, 
                                                                                          disparities_indexes, 
                                                                                          directions)
        return setpoint, target_steering_angle_rad

    def find_disparities(self, distances):
        """
        Find disparities between distances from lidar simulation

        Parameters
        -----------
        distances: list of float
            Distances as sent by the sensor. Their indices should be counted in the mathematical positive rotational sense.
            (Furthest right ray would be index 0)

        Returns
        -------
        disparities_indexes: list of int
            Indexes pointing out where in the distances list there are disparities.
            The indexes point to the shorter end of the disparity.
        directions: list of bool
            Each disparity has a direction in which the larger distance falls.
            True if distance left of the disparity point is larger and False otherwise.
        """
        disparities_indexes = []
        directions = []
        for i in range(len(distances) - 1):
            if abs(distances[i] - distances[i + 1]) >= self.disparity_threshold:
                if distances[i] < distances[i + 1]:
                    disparities_indexes.append(i)
                    directions.append(True)
                else:
                    disparities_indexes.append(i + 1)
                    directions.append(False)
        return disparities_indexes, directions

    def extend_disparities(self, distances, rays, disparities_indexes, directions):
        overwritten_indexes = []

        # Check only the rays going into the "open" direction
        for i, disparity_index in enumerate(disparities_indexes):
            if directions[i]:
                indices = range(disparity_index + 1, len(rays))
            else:
                indices = range(disparity_index - 1, -1, -1)

            # Vector from origin point to short side of disparity
            disp_vec = np.array(
                [
                    rays[disparity_index].xy[0][1] - rays[disparity_index].xy[0][0],
                    rays[disparity_index].xy[1][1] - rays[disparity_index].xy[1][0],
                ]
            )
            for j in indices:
                # Vector from origin point to intersection next to short side of disparity
                ray_vec = np.array(
                    [
                        rays[j].xy[0][1] - rays[j].xy[0][0],
                        rays[j].xy[1][1] - rays[j].xy[1][0],
                    ]
                )
                # Calculate angle between the rays

                if (rays[disparity_index].length * rays[j].length) == 0: #div by zero...
                    print("DIV BY ZERO!!!!!!!")
                    break

                angle = np.arccos(
                    np.dot(disp_vec, ray_vec)
                    / (rays[disparity_index].length * rays[j].length)
                )

                # Calculate distance to disparity point
                d = np.sin(angle) * distances[disparity_index]
                epsilon = 0.0001
                if d < (self.car_width / 2 + self.tolerance):
                    # Dont overwrite closer distances with further ones
                    if distances[j] <= distances[disparity_index]:
                        continue
                    distances[j] = distances[disparity_index] + j * epsilon
                    overwritten_indexes.append(j)
                else:
                    break
        

        max_distance_index = np.argmax(distances)

        if rays[max_distance_index].length == 0:
            return None, None, None

        t = distances[max_distance_index] / rays[max_distance_index].length
        setpoint = [
            (1 - t) * rays[max_distance_index].xy[0][0]
            + t * rays[max_distance_index].xy[0][1],
            (1 - t) * rays[max_distance_index].xy[1][0]
            + t * rays[max_distance_index].xy[1][1],
        ]
        angle = np.arctan2(
            setpoint[1] - rays[max_distance_index].xy[1][0],
            setpoint[0] - rays[max_distance_index].xy[0][0],
        )
        return distances, setpoint, angle
