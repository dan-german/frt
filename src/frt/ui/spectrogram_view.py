import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget


class SpectrogramPlot(PlotWidget):
    def __init__(self, y_label: str, y_units: str | None = None):
        super().__init__()
        self.setLabel("bottom", "Time", units="s")
        self.setLabel("left", y_label, units=y_units)

        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(pg.colormap.get("inferno"))
        self.addItem(self.image_item)

        self.overlay_item = pg.ImageItem()
        self.overlay_item.setColorMap(pg.colormap.get("viridis"))
        self.overlay_item.setOpacity(1.0)
        self.overlay_item.hide()
        self.addItem(self.overlay_item)

    def update_spectrogram(
        self,
        db: np.ndarray,
        duration: float,
        y_max: float,
    ) -> None:
        self.image_item.setImage(db.T)
        self.image_item.setRect(0, 0, max(duration, 1), max(y_max, 1))

    def update_overlay(
        self,
        db: np.ndarray,
        x_start: float,
        duration: float,
        y_max: float,
    ) -> None:
        if db.size == 0 or duration <= 0:
            self.clear_overlay()
            return

        self.overlay_item.setImage(db.T)
        self.overlay_item.setRect(x_start, 0, duration, max(y_max, 1))
        self.overlay_item.show()

    def clear_overlay(self) -> None:
        self.overlay_item.clear()
        self.overlay_item.hide()
