from collections.abc import Callable

import pyqtgraph as pg
from pyqtgraph import PlotWidget
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from PyQt6.QtWidgets import QGraphicsProxyWidget

from frt.config import PLOT_HEIGHT, WAVEFORM_WIDTH, WINDOW_WIDTH


class DragViewBox(pg.ViewBox):
    def __init__(
        self,
        sample_index: int,
        on_moved: Callable[[int, int], None],
    ):
        super().__init__()
        self.sample_index = sample_index
        self.on_moved = on_moved
        self.proxy: QGraphicsProxyWidget | None = None
        self._x0: float | None = None

    def bind(self, proxy: QGraphicsProxyWidget) -> None:
        self.proxy = proxy

    def mouseDragEvent(self, ev: MouseDragEvent, axis=None) -> None:
        ev.accept()
        if self.proxy is None:
            return

        if ev.isStart():
            self._x0 = self.proxy.pos().x()

        if self._x0 is None:
            return

        dx = ev.screenPos().x() - ev.buttonDownScreenPos().x()
        max_x = max(WINDOW_WIDTH - self.proxy.widget().width(), 0)
        new_x = min(max(self._x0 + dx, 0), max_x)

        self.on_moved(self.sample_index, int(new_x))
        self.proxy.setPos(new_x, self.proxy.pos().y())


def create_draggable_plot(
    data,
    sample_index: int,
    on_moved: Callable[[int, int], None],
) -> QGraphicsProxyWidget:
    drag_view_box = DragViewBox(sample_index, on_moved)

    plot_widget = PlotWidget(viewBox=drag_view_box)
    plot_widget.setDefaultPadding(0)
    plot_widget.plot(data)
    plot_widget.hideAxis("bottom")
    plot_widget.hideAxis("left")
    plot_widget.setMouseEnabled(x=False, y=False)
    plot_widget.setFixedSize(WAVEFORM_WIDTH, PLOT_HEIGHT)

    proxy = QGraphicsProxyWidget()
    proxy.setWidget(plot_widget)
    proxy.setPos(0, sample_index * PLOT_HEIGHT)

    drag_view_box.bind(proxy)
    return proxy

