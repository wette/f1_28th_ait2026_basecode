from .track import Track
import numpy as np
import shapely

class LidarSensor:
    def __init__(self, track: Track, field_of_view_deg: float, numRays: int, rayLength_px: int):
        self.track = track
        self.fov_rad = np.deg2rad(field_of_view_deg)
        self.numRays = numRays
        self.rayLength = rayLength_px

    def __generateRays(self, xPos:int, yPos:int, angle_rad:float):
        rays = []
        ray_angle = angle_rad - self.fov_rad / 2
        for i in range(self.numRays):
            angle_rad = ray_angle
            delta_x = np.cos(angle_rad) * self.rayLength
            delta_y = np.sin(angle_rad) * self.rayLength
            rays.append(
                shapely.LineString(
                    [
                        [xPos, yPos],
                        [xPos + delta_x, yPos + delta_y],
                    ]
                )
            )
            ray_angle += self.fov_rad / (self.numRays - 1)

        return rays
    
    def __computeIntersectionsWithTrack(self, rays, x, y, opponents : list['Vehicle'] = []):
        distances = []
        vehicleLocation = shapely.Point(int(x), int(y))
        intersectedRays = []

        obstacles = [self.track.innerRing, self.track.outerRing]
        for v in opponents:
            bbox = shapely.Polygon(v.getBoundingBox())
            if not bbox.contains(vehicleLocation):
                obstacles.append(bbox)
            bbox = shapely.Polygon(v.getBoundingBox(at_time=v.last_update+0.1, use_command_history=False))
            if not bbox.contains(vehicleLocation):
                obstacles.append(bbox)

        for ray in rays:
            dist = []
            intray = []
            for bound in obstacles:
                intersection = bound.intersection(ray)
                if intersection.is_empty:
                    dist.append(self.rayLength)
                    intray.append(ray)
                else:
                    d = int(shapely.distance(vehicleLocation, intersection))
                    dist.append(d)
                    
                    point = ray.interpolate(d)
                    intray.append(shapely.LineString([vehicleLocation, point]))
            
            min_index = np.argmin(dist)
            distances.append(dist[min_index])
            intersectedRays.append(intray[min_index])

        return distances, intersectedRays

    #Compute Lidar Rays and length w.r.t. the racetrack.
    # Inputs:  xPos, yPos: Position of the vehicle
    #          angle:      yaw-angle of the vehicle
    #          opponents: other vehicles which can be detected by the lidar
    #
    # Outputs: distances:           distance readings w.r.t. the racetrack -> list of floats
    #          intersectedRays:     the rays sent out by the lidar -> list of shapely.LineString objects
    def getReadings(self, xPos:int, yPos:int, angle:float, opponents : list['Vehicle'] = []):
        rays = self.__generateRays(xPos, yPos, angle)
        distances, intersectedRays = self.__computeIntersectionsWithTrack(rays, xPos, yPos, opponents)

        return distances, intersectedRays




