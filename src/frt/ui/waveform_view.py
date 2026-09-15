from collections.abc import Callable

import pyqtgraph as pg
from pyqtgraph import PlotWidget
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsProxyWidget, QMenu

from frt.audio.timeline import NoteEvent
from frt.config import MIN_NOTE_WIDTH, PLOT_HEIGHT, TIMELINE_SAMPLES_PER_PIXEL


class DragViewBox(pg.ViewBox):
    def __init__(
        self,
        event_id: int,
        on_moved: Callable[[int, int], int],
    ):
        super().__init__()
        self.event_id = event_id
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

        requested_x = max(
            self._x0 + ev.screenPos().x() - ev.buttonDownScreenPos().x(),
            0,
        )
        accepted_x = self.on_moved(self.event_id, int(requested_x))
        self.proxy.setPos(accepted_x, self.proxy.pos().y())


class WaveformPlotWidget(PlotWidget):
    def __init__(
        self,
        event: NoteEvent,
        data,
        drag_view_box: DragViewBox,
        selected: bool,
        on_selected: Callable[[int], None],
        on_toggled: Callable[[int], None],
        on_deleted: Callable[[int], None],
    ):
        super().__init__(viewBox=drag_view_box)
        self.event = event
        self.on_selected = on_selected
        self.on_toggled = on_toggled
        self.on_deleted = on_deleted

        self.setDefaultPadding(0)
        self.plot(data)
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.setMouseEnabled(x=False, y=False)
        width = max(
            int(event.duration_samples / TIMELINE_SAMPLES_PER_PIXEL),
            MIN_NOTE_WIDTH,
        )
        self.setFixedSize(width, PLOT_HEIGHT - 4)
        self.set_selected(selected)

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(
            "border: 2px solid #65d4ff;" if selected else "border: 0px;"
        )

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_selected(self.event.id)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu()
        mute_action = menu.addAction("Unmute" if not self.event.enabled else "Mute")
        delete_action = menu.addAction("Delete")
        action = menu.exec(event.globalPos())

        if action == mute_action:
            self.on_toggled(self.event.id)
        elif action == delete_action:
            self.on_deleted(self.event.id)


class NoteProxyWidget(QGraphicsProxyWidget):
    def __init__(
        self,
        event: NoteEvent,
        data,
        selected: bool,
        on_moved: Callable[[int, int], int],
        on_selected: Callable[[int], None],
        on_toggled: Callable[[int], None],
        on_deleted: Callable[[int], None],
    ):
        super().__init__()
        drag_view_box = DragViewBox(event.id, on_moved)
        self.setWidget(
            WaveformPlotWidget(
                event,
                data,
                drag_view_box,
                selected,
                on_selected,
                on_toggled,
                on_deleted,
            )
        )
        drag_view_box.bind(self)
        self.setData(0, "note")
        self.setData(1, event.id)
        self.setOpacity(1.0 if event.enabled else 0.35)
        self.setPos(
            event.start_sample / TIMELINE_SAMPLES_PER_PIXEL,
            event.string_index * PLOT_HEIGHT + 2,
        )


def create_note_item(
    event: NoteEvent,
    data,
    selected: bool,
    on_moved: Callable[[int, int], int],
    on_selected: Callable[[int], None],
    on_toggled: Callable[[int], None],
    on_deleted: Callable[[int], None],
) -> QGraphicsProxyWidget:
    return NoteProxyWidget(
        event,
        data,
        selected,
        on_moved,
        on_selected,
        on_toggled,
        on_deleted,
    )
