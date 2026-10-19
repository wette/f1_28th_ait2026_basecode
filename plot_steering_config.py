# -*- coding: utf-8 -*-
"""
Interaktiver Editor für Messdaten (y-Werte per Maus verschieben) mit matplotlib.
- Linksklick nahe Punkt: auswählen & ziehen (nur y wird geändert)
- Rechtsklick: Punkt abwählen
- Taste 'c': Daten als Python-Liste in die Konsole ausgeben (Copy & Paste)
- Taste 's': Daten als JSON in die Konsole ausgeben
- Taste 'r': auf Originaldaten zurücksetzen
- Taste 'q' oder ESC: schließen

Voraussetzung: matplotlib
    pip install matplotlib
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt




print("Backend:", matplotlib.get_backend())


# ----------------------- Ihre Daten hier einsetzen -----------------------
data: List[Tuple[float, float]] = [(10, -47.0), (20, -45.0), (30, -40.0), (40, -35.0), (50, -28.0), (60, -23.0), (70, -19.0), (80, -10.0), (90, -4.0), (100, 3.0), (110, 9.0), (120, 20.0), (130, 27.0), (140, 34.0), (150, 38.0), (160, 44.0), (170, 50.0)]

data = [(10, 41.47479972843025), (50, 25.66474028449667), (90, 5.179684238656864), (130, -10.119637971863794), (180, -24.67991246626858)]
# -----------------------------------------------------------------------


@dataclass
class DragState:
    idx: Optional[int] = None
    dragging: bool = False


class InteractiveYEditor:
    def __init__(self, data_xy: List[Tuple[float, float]]):
        self.original = [(float(x), float(y)) for x, y in data_xy]
        self.data = [(float(x), float(y)) for x, y in data_xy]
        self.state = DragState()

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title("HSBIKI - Interaktiver Messdaten-Editor (matplotlib)")

        self.xs = [x for x, _ in self.data]
        self.ys = [y for _, y in self.data]

        # Punkte + Linie
        self.line, = self.ax.plot(self.xs, self.ys, "-", lw=1.5, alpha=0.7)
        self.sc = self.ax.scatter(self.xs, self.ys, s=50, picker=False)

        # Hervorhebung des aktiven Punktes
        self.active_sc = self.ax.scatter([], [], s=120, facecolors="none", edgecolors="red", linewidths=2)

        self.status = self.ax.text(
            0.01, 0.99,
            "LMB: Punkt ziehen (nur y) | c: Copy&Paste | r: Reset | q/ESC: Quit",
            transform=self.ax.transAxes,
            va="top", ha="left"
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.3)

        # Event-Handler
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._autoscale_with_margin()

    def _autoscale_with_margin(self):
        xs = [x for x, _ in self.data]
        ys = [y for _, y in self.data]
        if not xs or not ys:
            return
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        dx = (x_max - x_min) or 1.0
        dy = (y_max - y_min) or 1.0
        self.ax.set_xlim(x_min - 0.05 * dx, x_max + 0.05 * dx)
        self.ax.set_ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)

    def _nearest_point_index(self, x: float, y: float) -> Optional[int]:
        """Findet den nächsten Punkt in *Display*-Koordinaten (pixel), robust gegen Achsenskalierung."""
        if x is None or y is None:
            return None

        # Transformiere Datenpunkte in Displaykoordinaten
        disp = self.ax.transData.transform
        px, py = disp((x, y))

        best_i = None
        best_d2 = float("inf")
        for i, (xi, yi) in enumerate(self.data):
            qx, qy = disp((xi, yi))
            d2 = (qx - px) ** 2 + (qy - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        # Schwellwert in Pixeln: wie nah muss der Klick am Punkt sein?
        threshold_px = 12.0
        if best_d2 <= threshold_px ** 2:
            return best_i
        return None

    def _update_artists(self):
        self.xs = [x for x, _ in self.data]
        self.ys = [y for _, y in self.data]

        self.line.set_data(self.xs, self.ys)
        self.sc.set_offsets(list(zip(self.xs, self.ys)))

        if self.state.idx is None:
            self.active_sc.set_offsets([])
        else:
            xi, yi = self.data[self.state.idx]
            self.active_sc.set_offsets([(xi, yi)])

        self.fig.canvas.draw_idle()

    def _print_as_python(self):
        # Schön formatiert, gut für Copy & Paste
        s = "data = [\n" + ",\n".join(f"    ({x:g}, {y:.6g})" for x, y in self.data) + "]\n"
        print(s)

    def _print_as_json(self):
        print(json.dumps(self.data, indent=2))

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        # Rechtsklick: abwählen
        if event.button == 3:
            self.state.idx = None
            self.state.dragging = False
            self._update_artists()
            return

        # Linksklick: auswählen
        if event.button == 1:
            idx = self._nearest_point_index(event.xdata, event.ydata)
            if idx is not None:
                self.state.idx = idx
                self.state.dragging = True
                self._update_artists()

    def on_motion(self, event):
        if not self.state.dragging or self.state.idx is None:
            return
        if event.inaxes != self.ax:
            return
        if event.ydata is None:
            return

        # Nur y ändern, x bleibt fix
        x_fixed, _ = self.data[self.state.idx]
        self.data[self.state.idx] = (x_fixed, float(event.ydata))
        self._update_artists()

    def on_release(self, event):
        if event.button == 1:
            self.state.dragging = False

    def on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
            return
        if event.key == "c":
            self._print_as_python()
            return
        if event.key == "s":
            self._print_as_json()
            return
        if event.key == "r":
            self.data = [(x, y) for x, y in self.original]
            self.state.idx = None
            self.state.dragging = False
            self._update_artists()
            return

    def show(self):
        plt.show()


if __name__ == "__main__":
    editor = InteractiveYEditor(data)
    editor.show()