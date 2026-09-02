import sys
import numpy as np
import pyqtgraph as pg
from sample_builder import *
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QPushButton, QVBoxLayout, QWidget
)
from PyQt6.QtCore import Qt
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
import librosa

WINDOW_WIDTH = 1200
WIDTH_HEIGHT = 900
PLOT_HEIGHT = 40

builder = SampleBuilder()
sample = builder.build((0, 0, None, None, None, None))


class DragVB(pg.ViewBox):
    def bind(self, proxy):
        self.proxy, self._x0 = proxy, None

    def mouseDragEvent(self, ev: MouseDragEvent, axis=None):
        # print(ev..y())
        ev.accept()
        if ev.isStart():
            self._x0 = self.proxy.pos().x()
        dx = ev.screenPos().x() - ev.buttonDownScreenPos().x()
        new_x = min(max(self._x0 + dx, 0),
                    max(WINDOW_WIDTH - self.proxy.widget().width(), 0))
        sample.reposition(0, int(new_x))
        self.proxy.setPos(new_x, self.proxy.pos().y())


def create_draggable_plot(data, y: int):
    drag_view_box = DragVB()

    plot_widget = pg.PlotWidget(viewBox=drag_view_box)
    plot_widget.plot(data)
    plot_widget.hideAxis('bottom')
    plot_widget.hideAxis('left')
    plot_widget.setMouseEnabled(x=False, y=False)
    plot_widget.setFixedSize(1000, 40)

    proxy = QGraphicsProxyWidget()
    proxy.setWidget(plot_widget)
    proxy.setPos(0, y)

    drag_view_box.bind(proxy)
    return proxy


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(WINDOW_WIDTH, WIDTH_HEIGHT)
        container = QWidget()
        button = QPushButton("Play")
        button.clicked.connect(sample.play)

        scene = QGraphicsScene()
        scene.setBackgroundBrush(Qt.GlobalColor.darkGray)
        for i, note_sample in enumerate(sample.canvas):
            scene.addItem(create_draggable_plot(note_sample, i * PLOT_HEIGHT))

        view = QGraphicsView(scene)
        view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view)
        layout.addWidget(button)

        self.setCentralWidget(container)


        # scene.setSceneRect(0, 0, WINDOW_WIDTH, PLOT_HEIGHT * 4)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
