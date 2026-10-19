import cv2 as cv
import math
import pickle
from shapely.geometry import Point, Polygon, LinearRing
import numpy as np

class Track:
    def __init__(self):

        # coordinates of polygon represending the left, resp. right track borders.
        self.outerBounds = []
        self.innerBounds = []

        # polygon objects from shapely.
        self.outerRing = None
        self.innerRing = None
        self.polygon = None

        self.img = None            # current displayed image (overlay)
        self.img_base = None       # original image to redraw from

        self.mode = None

        # --- NEW: drag state ---
        self.dragging = False
        self.drag_index = None       # index of point being dragged
        self.drag_list_name = None   # "outer" or "inner"
        self.pick_radius = 12        # pixel radius to select a point

        # --- NEW: polyline drawing options ---
        self.draw_polylines = True
        self.close_polylines = False   # True: letzter Punkt wird mit erstem verbunden

    def pointInsideTrack(self, point: Point):
        return self.polygon.contains(point)

    def saveToFile(self, filename):
        with open(filename, 'wb') as file:
            track = dict()
            track['outerBounds'] = self.outerBounds
            track['innerBounds'] = self.innerBounds
            pickle.dump(track, file)

    def loadFromFile(self, filename):
        with open(filename, 'rb') as file:
            track = pickle.load(file)
            self.outerBounds = track['outerBounds']
            self.innerBounds = track['innerBounds']
            self.convertToPolygon()

    def convertToPolygon(self):
        try:
            self.innerRing = LinearRing(self.innerBounds)
            self.outerRing = LinearRing(self.outerBounds)
        except Exception as e:
            print(f"Failure generating linear ring: {e}")

        try:
            self.polygon = Polygon(shell=self.outerRing, holes=[self.innerRing])
        except Exception as e:
            print(f"Failure generating polygon: {e}")

    # -------------- NEW helper methods --------------

    def _get_active_list(self):
        """Return currently active list for new point creation."""
        return self.outerBounds if self.mode == "outer" else self.innerBounds

    def _find_nearest_point(self, x, y):
        """
        Find nearest point within pick_radius among BOTH lists.
        Returns: (list_name, index) or (None, None)
        """
        best = (None, None)
        best_d2 = self.pick_radius * self.pick_radius

        for name, pts in (("outer", self.outerBounds), ("inner", self.innerBounds)):
            for i, (px, py) in enumerate(pts):
                d2 = (px - x) ** 2 + (py - y) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (name, i)

        return best

    def _draw_points_overlay(self):
        """
        Redraw overlay image from base image + all points + connecting lines.
        """
        if self.img_base is None:
            return

        self.img = self.img_base.copy()

        # ---- draw polylines first (so points are on top) ----
        if self.draw_polylines:
            # outer polyline (red)
            if len(self.outerBounds) >= 2:
                pts = [(int(x), int(y)) for x, y in self.outerBounds]
                cv.polylines(
                    self.img,
                    [cv.UMat(cv.UMat.get(0))] if False else [ # no-op, keeps style minimal
                        np.array(pts, dtype=np.int32).reshape((-1, 1, 2)) ],
                    isClosed=self.close_polylines,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv.LINE_AA
                )

            # inner polyline (green)
            if len(self.innerBounds) >= 2:
                pts = [(int(x), int(y)) for x, y in self.innerBounds]
                cv.polylines(
                    self.img,
                    [ np.array(pts, dtype=np.int32).reshape((-1, 1, 2)) ],
                    isClosed=self.close_polylines,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv.LINE_AA
                )

        # ---- draw points ----
        for (x, y) in self.outerBounds:
            cv.circle(self.img, (int(x), int(y)), 5, (0, 0, 255), -1)

        for (x, y) in self.innerBounds:
            cv.circle(self.img, (int(x), int(y)), 5, (0, 255, 0), -1)

        # ---- highlight selected point ----
        if self.dragging and self.drag_list_name is not None and self.drag_index is not None:
            pts = self.outerBounds if self.drag_list_name == "outer" else self.innerBounds
            if 0 <= self.drag_index < len(pts):
                x, y = pts[self.drag_index]
                cv.circle(self.img, (int(x), int(y)), 9, (255, 255, 0), 2)

    # -------------- UPDATED mouse callback --------------

    def mouseCallback(self, event, x, y, flags, param):
        """
        - Click on empty space: create a new point in current mode (outer/inner).
        - Click on existing point: start dragging it, move with mouse, release to drop.
        - Optional: SHIFT + click on point: delete point.
        """

        # SHIFT-delete (optional)
        if event == cv.EVENT_LBUTTONDOWN and (flags & cv.EVENT_FLAG_SHIFTKEY):
            name, idx = self._find_nearest_point(x, y)
            if name is not None:
                pts = self.outerBounds if name == "outer" else self.innerBounds
                pts.pop(idx)
                self._draw_points_overlay()
            return

        if event == cv.EVENT_LBUTTONDOWN:
            # If click is near an existing point -> start dragging it
            name, idx = self._find_nearest_point(x, y)
            if name is not None:
                self.dragging = True
                self.drag_list_name = name
                self.drag_index = idx
                self._draw_points_overlay()
                return

            # Otherwise: add new point to current mode
            pts = self._get_active_list()
            pts.append((x, y))
            self._draw_points_overlay()
            return

        if event == cv.EVENT_MOUSEMOVE:
            if self.dragging and self.drag_list_name is not None and self.drag_index is not None:
                pts = self.outerBounds if self.drag_list_name == "outer" else self.innerBounds
                if 0 <= self.drag_index < len(pts):
                    pts[self.drag_index] = (x, y)
                    self._draw_points_overlay()
            return

        if event == cv.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                self.drag_list_name = None
                self.drag_index = None
                self._draw_points_overlay()
            return

    def manuallyPickTrackBorders(self, img):

        print(
            "Mouse-leftclick to set controlpoints for the track boundaries.\n"
            "Sie können vorhandene Punkte per Linksklick greifen und mit gedrückter Taste verschieben.\n"
            "SHIFT + Linksklick löscht optional einen Punkt.\n"
            "First, select the OUTER outline of the track. Then switch to the inner boundary.\n"
            "press n to change mode to inner bounding\n"
            "press b to return to mode outer bounding\n"
            "press q to finish the process.\n"
        )

        windowName = "Pick Track Borders"
        cv.namedWindow(windowName, cv.WINDOW_NORMAL)
        cv.setMouseCallback(windowName, self.mouseCallback)

        self.mode = "outer"

        # base image for redraw, do not draw permanently into it
        self.img_base = img.copy()
        self._draw_points_overlay()

        while True:
            cv.imshow(windowName, self.img)  # show overlay image
            key = cv.waitKey(1) & 0xFF

            if key == ord("n") and self.mode == "outer":
                self.mode = "inner"
                print("Switched to inner outline.")

            elif key == ord("b") and self.mode != "outer":
                self.mode = "outer"
                print("Switched to outer outline.")

            elif key == ord("q"):
                break

        cv.destroyWindow(windowName)

        # convert points to polygon
        self.convertToPolygon()