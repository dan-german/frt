"""
Minimal pyqtgraph real-time plot example.

Install first:
    pip install pyqtgraph pyqt6 numpy

Run:
    python test_pyqtgraph.py

This just plots a sine wave whose frequency slowly changes,
updated live via a QTimer -- a stand-in for how you'd later
push FFT frames onto the same kind of plot.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# --- window setup ---
app = pg.mkQApp("pyqtgraph test")
win = pg.GraphicsLayoutWidget(show=True, title="Live plot test")
win.resize(800, 400)

plot = win.addPlot(title="Live sine wave")
plot.setYRange(-1.2, 1.2)
curve = plot.plot(pen='y')

# --- data ---
x = np.linspace(0, 4 * np.pi, 1000)
state = {"freq": 1.0}

def update():
    state["freq"] += 0.02
    y = np.sin(x * state["freq"])
    curve.setData(x, y)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30//4)  # ms, ~33 fps

if __name__ == "__main__":
    pg.exec()