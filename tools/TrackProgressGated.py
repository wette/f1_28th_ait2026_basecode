from shapely.geometry import LinearRing, LineString, Point, Polygon
import numpy as np
from shapely import STRtree

# ---------- Kern: progress mit Gates ----------
class TrackProgressGated:

    # ---------- Hilfsfunktionen ----------
    @staticmethod
    def _ring_to_linestring(r: LinearRing) -> LineString:
        return LineString(r.coords)

    @staticmethod
    def _resample_equally(line: LineString, n: int) -> np.ndarray:
        L = line.length
        d = np.linspace(0.0, L, n, endpoint=False)
        pts = [line.interpolate(di) for di in d]
        return np.array([[p.x, p.y] for p in pts], dtype=float)

    @staticmethod
    def _rotate_arrays_to_start(center, left, right, start_xy):
        """
        Rotiert die diskreten Arrays so, dass Index 0 beim projizierten Startpunkt liegt.
        Das ist diskret (auf Sample-Basis), aber stabil bei genügend n_samples.
        """
        center_ls = LineString(np.vstack([center, center[0]]))
        s0 = center_ls.project(Point(start_xy))
        p0 = center_ls.interpolate(s0)
        p0_xy = np.array([p0.x, p0.y])

        # nächster Samplepunkt zur Startprojektion
        i0 = int(np.argmin(np.sum((center - p0_xy) ** 2, axis=1)))

        def rot(a):
            return np.vstack([a[i0:], a[:i0]])

        center_r = rot(center)
        left_r   = rot(left)
        right_r  = rot(right)

        # Setzen wir den ersten Center-Punkt exakt auf die Projektion (optional, aber gut)
        center_r[0] = p0_xy
        return center_r, left_r, right_r

    @staticmethod
    def _segment_projection_t(p, a, b):
        """Projektionsparameter t von Punkt p auf Segment a->b, geklemmt auf [0,1]."""
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom == 0.0:
            return 0.0
        t = float(np.dot(p - a, ab) / denom)
        return max(0.0, min(1.0, t))

    def __init__(self, left_ring: LinearRing, right_ring: LinearRing,
                 start_xy: tuple[float, float], n_samples: int = 6000):
        self.n = int(n_samples)
        left_ls = TrackProgressGated._ring_to_linestring(left_ring)
        right_ls = TrackProgressGated._ring_to_linestring(right_ring)

        self.left = TrackProgressGated._resample_equally(left_ls, self.n)
        self.right = TrackProgressGated._resample_equally(right_ls, self.n)
        self.center = 0.5 * (self.left + self.right)

        # auf Start rotieren
        self.center, self.left, self.right = TrackProgressGated._rotate_arrays_to_start(
            self.center, self.left, self.right, start_xy
        )

        # Vorberechnung: Segmentlängen und kumulative s
        c_next = np.roll(self.center, -1, axis=0)
        seg_vec = c_next - self.center
        self.seg_len = np.linalg.norm(seg_vec, axis=1)
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.seg_len[:-1])])
        self.track_length = float(np.sum(self.seg_len))

        # Gates als Polygone (Quadrilaterale): left[i]-left[i+1]-right[i+1]-right[i]
        gates = []
        for i in range(self.n):
            i2 = (i + 1) % self.n
            poly = Polygon([tuple(self.left[i]),
                            tuple(self.left[i2]),
                            tuple(self.right[i2]),
                            tuple(self.right[i])])
            gates.append(poly)
        self.gates = gates

        # Spatial Index für schnelle Kandidatensuche
        self._tree = STRtree(self.gates)
        self._gate_geoms = self.gates  # alias



    def progress(self,
             vehicle_xy: tuple[float, float],
             prev_progress: float | None = None,
             yaw: float | None = None,
             yaw_in_degrees: bool = False,
             search_radius: float = 15.0,
             max_jump_frac: float = 0.05,
             yaw_weight: float = 5.0) -> dict:
        """
        yaw: Fahrzeug-Yaw (Standard: Radiant). Wenn yaw_in_degrees=True, wird Grad->Radiant umgerechnet.
        yaw_weight: Gewichtung des Yaw-Terms im Score (größer = stärkerer Einfluss).

        Hinweis: Durch die verwendete Angle-Cost ist Fahren "falsch herum" erlaubt
        (antiparallel zählt als gut).
        """
        def _wrap_to_pi(a: float) -> float:
            return (a + np.pi) % (2.0 * np.pi) - np.pi

        def _angle_diff(a: float, b: float) -> float:
            """kleinste Winkeldifferenz a-b in [-pi, pi]."""
            return _wrap_to_pi(a - b)

        def _segment_heading(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
            v = b_xy - a_xy
            return float(np.arctan2(v[1], v[0]))

        def _angle_cost(yaw: float, seg_heading: float) -> float:
            """
            Kosten für Ausrichtung:
            - 0 wenn yaw parallel ODER antiparallel zum Segment ist (vorwärts oder rückwärts fahren erlaubt)
            - maximal bei 90°
            """
            d = abs(_angle_diff(yaw, seg_heading))          # 0..pi
            d = min(d, abs(np.pi - d))                      # 0..pi/2 (antiparallel wird wie parallel)
            return float(d / (0.5 * np.pi))                 # 0..1
        
        p = Point(vehicle_xy)
        p_xy = np.array(vehicle_xy, dtype=float)

        yaw_rad = None
        if yaw is not None:
            yaw_rad = float(np.deg2rad(yaw) if yaw_in_degrees else yaw)

        # Kandidaten: Gates nahe Fahrzeug
        query_geom = p.buffer(search_radius)
        cand = self._tree.query(query_geom)

        best = None  # (score, i, t, proj_xy)

        for poly in cand:
            i = int(poly)  # bei Ihnen liefert STRtree offenbar Indizes
            poly = self._gate_geoms[i]

            a = self.center[i]
            b = self.center[(i + 1) % self.n]

            t = TrackProgressGated._segment_projection_t(p_xy, a, b)
            proj = a + t * (b - a)
            dist = float(np.linalg.norm(p_xy - proj))

            inside = poly.contains(p) or poly.distance(p) < 0.2

            # Basis-Score: inside gewinnt, dist als Feinkriterium
            score = (0 if inside else 1) * 1e6 + dist

            # Yaw-Term: bevorzugt Segmente, deren Richtung zur Fahrzeugausrichtung passt
            # (parallel ODER antiparallel sind beide gut -> falsch herum möglich)
            if yaw_rad is not None:
                seg_h = _segment_heading(a, b)
                score += yaw_weight * _angle_cost(yaw_rad, seg_h)

            if best is None or score < best[0]:
                best = (score, i, t, proj)

        # Fallback, falls nichts gefunden
        if best is None:
            best = (1e18, 0, 0.0, self.center[0])
            for i in range(self.n):
                a = self.center[i]
                b = self.center[(i + 1) % self.n]
                t = TrackProgressGated._segment_projection_t(p_xy, a, b)
                proj = a + t * (b - a)
                dist = float(np.linalg.norm(p_xy - proj))

                score = dist
                if yaw_rad is not None:
                    seg_h = _segment_heading(a, b)
                    score += yaw_weight * _angle_cost(yaw_rad, seg_h)

                if score < best[0]:
                    best = (score, i, t, proj)

        _, i_best, t_best, proj_xy = best

        s = float(self.cum_s[i_best] + t_best * self.seg_len[i_best])
        frac = (s / self.track_length) if self.track_length > 0 else 0.0

        # Stabilisierung mit prev_progress (optional)
        if prev_progress is not None:
            cands = np.array([frac, frac + 1.0, frac - 1.0])
            frac = float(cands[np.argmin(np.abs(cands - prev_progress))])
            delta = float(np.clip(frac - prev_progress, -max_jump_frac, max_jump_frac))
            frac = (prev_progress + delta) % 1.0
            s = frac * self.track_length

        return {
            "progress_frac": frac,
            "progress_percent": 100.0 * frac,
            "s": s,
            "track_length": self.track_length,
            "projected_point": Point(float(proj_xy[0]), float(proj_xy[1])),
            "gate_index": int(i_best),
        }