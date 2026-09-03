from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from frt.audio.analysis import compute_cqt, compute_stft
from frt.audio.sample_batch import SampleBatch
from frt.config import (
    ANALYSIS_UPDATE_DELAY_MS,
    PLOT_HEIGHT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from frt.ui.spectrogram_view import SpectrogramPlot
from frt.ui.waveform_view import create_draggable_plot


class MainWindow(QMainWindow):
    def __init__(self, sample: SampleBatch):
        super().__init__()
        self.sample = sample
        self.stft_plot = SpectrogramPlot("Frequency", "Hz")
        self.cqt_plot = SpectrogramPlot("Frequency (log)")

        self.analysis_update_timer = QTimer(self)
        self.analysis_update_timer.setSingleShot(True)
        self.analysis_update_timer.timeout.connect(self.update_spectrograms)

        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setCentralWidget(self.create_container())
        self.update_spectrograms()

    def create_container(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.create_top_row())
        layout.addWidget(self.create_plots_view())
        layout.addWidget(self.create_play_button())
        return container

    def create_top_row(self) -> QWidget:
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.addWidget(self.stft_plot)
        top_layout.addWidget(self.cqt_plot)
        return top_row

    def create_plots_view(self) -> QGraphicsView:
        plots_scene = QGraphicsScene()
        plots_scene.setBackgroundBrush(Qt.GlobalColor.darkGray)

        for sample_index, note_sample in enumerate(self.sample.canvas):
            plots_scene.addItem(
                create_draggable_plot(
                    note_sample,
                    sample_index,
                    self.on_waveform_moved,
                )
            )

        plots_scene.setSceneRect(
            0,
            0,
            WINDOW_WIDTH,
            (len(self.sample.canvas) + 1) * PLOT_HEIGHT + 50,
        )

        plots_view = QGraphicsView(plots_scene)
        plots_view.setStyleSheet("border: 0px; background: transparent;")
        plots_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        plots_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        plots_view.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        return plots_view

    def create_play_button(self) -> QPushButton:
        play_button = QPushButton("Play")
        play_button.clicked.connect(self.sample.play)
        return play_button

    def on_waveform_moved(self, sample_index: int, x_position: int) -> None:
        self.sample.set_position_from_pixels(sample_index, x_position)
        self.analysis_update_timer.start(ANALYSIS_UPDATE_DELAY_MS)

    def update_spectrograms(self) -> None:
        audio = self.sample.combine()

        stft_db, stft_times, stft_freqs = compute_stft(audio, self.sample.sample_rate)
        self.stft_plot.update_spectrogram(
            stft_db,
            stft_times[-1] if len(stft_times) else 1,
            stft_freqs[-1] if len(stft_freqs) else 1,
        )

        cqt_db, cqt_times = compute_cqt(audio, self.sample.sample_rate)
        self.cqt_plot.update_spectrogram(
            cqt_db,
            cqt_times[-1] if len(cqt_times) else 1,
            cqt_db.shape[0] if cqt_db.ndim else 1,
        )

