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

    def update_spectrogram(
        self,
        db: np.ndarray,
        duration: float,
        y_max: float,
    ) -> None:
        self.image_item.setImage(db.T)
        self.image_item.setRect(0, 0, max(duration, 1), max(y_max, 1))

