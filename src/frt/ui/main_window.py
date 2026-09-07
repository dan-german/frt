from collections.abc import Iterable
from enum import Enum

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from frt.audio.analysis import (
    compute_cqt,
    compute_log_bins,
    compute_stft,
    max_cqt_bins,
    shift_frequency_bins,
)
from frt.audio.timeline import Timeline
from frt.config import (
    ANALYSIS_UPDATE_DELAY_MS,
    FRETS_PER_STRING,
    NUM_STRINGS,
    PLOT_HEIGHT,
    SAMPLE_RATE,
    TIMELINE_SAMPLES_PER_PIXEL,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from frt.ui.spectrogram_view import SpectrogramPlot
from frt.ui.waveform_view import create_note_item


class AnalysisPlot(Enum):
    STFT = "stft"
    CQT = "cqt"
    LOG_BINS = "log_bins"


DEFAULT_ANALYSIS_PLOTS = (
    AnalysisPlot.STFT,
    AnalysisPlot.CQT,
    AnalysisPlot.LOG_BINS,
)
CQT_SHIFT_MIN_SEMITONES = -24
CQT_SHIFT_MAX_SEMITONES = 24

type AnalysisPlotSelection = AnalysisPlot | Iterable[AnalysisPlot]


class TimelineScene(QGraphicsScene):
    def __init__(self, on_empty_double_clicked):
        super().__init__()
        self.on_empty_double_clicked = on_empty_double_clicked

    def mouseDoubleClickEvent(self, event) -> None:
        note_items = [
            item for item in self.items(event.scenePos()) if item.data(0) == "note"
        ]
        if note_items:
            super().mouseDoubleClickEvent(event)
            return

        string_index = int(event.scenePos().y() // PLOT_HEIGHT)
        if 0 <= string_index < NUM_STRINGS:
            start_sample = int(event.scenePos().x() * TIMELINE_SAMPLES_PER_PIXEL)
            self.on_empty_double_clicked(string_index, max(start_sample, 0))
            event.accept()


class MainWindow(QMainWindow):
    def __init__(
        self,
        timeline: Timeline,
        plots: AnalysisPlotSelection | None = None,
    ):
        super().__init__()
        self.timeline = timeline
        self.note_items = []
        self.analysis_plots = (
            DEFAULT_ANALYSIS_PLOTS
            if plots is None
            else (plots,)
            if isinstance(plots, AnalysisPlot)
            else tuple(plots)
        )
        self.spectrogram_plots = {
            plot: self.create_spectrogram_plot(plot) for plot in self.analysis_plots
        }
        self.selected_event_id: int | None = None
        self.selected_shift_semitones = 0
        self.fret_input = QSpinBox()
        self.technique_input = QSpinBox()
        self.shift_label = QLabel()

        self.analysis_update_timer = QTimer(self)
        self.analysis_update_timer.setSingleShot(True)
        self.analysis_update_timer.timeout.connect(self.update_spectrograms)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setCentralWidget(self.create_container())
        self.update_shift_label()
        self.refresh_note_items()
        self.update_spectrograms()

    def create_container(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.create_top_row())
        layout.addWidget(self.create_plots_view())
        layout.addWidget(self.create_controls())
        return container

    def create_top_row(self) -> QWidget:
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        for plot in self.analysis_plots:
            top_layout.addWidget(self.spectrogram_plots[plot])
        return top_row

    def create_spectrogram_plot(self, plot: AnalysisPlot) -> SpectrogramPlot:
        if plot is AnalysisPlot.STFT:
            return SpectrogramPlot("Frequency", "Hz")
        if plot is AnalysisPlot.CQT:
            return SpectrogramPlot("Frequency (log)")
        if plot is AnalysisPlot.LOG_BINS:
            return SpectrogramPlot("Log bins")
        raise ValueError(f"unsupported analysis plot: {plot}")

    def create_plots_view(self) -> QGraphicsView:
        self.plots_scene = TimelineScene(self.add_note_at)
        self.plots_scene.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.add_lane_backgrounds()
        self.update_scene_rect()

        self.plots_view = QGraphicsView(self.plots_scene)
        self.plots_view.setStyleSheet("border: 0px; background: transparent;")
        self.plots_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.plots_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plots_view.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.plots_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return self.plots_view

    def add_lane_backgrounds(self) -> None:
        pen = QPen(QColor("#555555"))
        for string_index in range(NUM_STRINGS):
            color = "#303030" if string_index % 2 == 0 else "#383838"
            rect = self.plots_scene.addRect(
                0,
                string_index * PLOT_HEIGHT,
                WINDOW_WIDTH,
                PLOT_HEIGHT,
                pen,
                QBrush(QColor(color)),
            )
            rect.setData(0, "lane")

    def create_controls(self) -> QWidget:
        controls = QWidget()
        layout = QHBoxLayout(controls)

        self.fret_input.setRange(0, FRETS_PER_STRING - 1)
        self.technique_input.setRange(
            0,
            max(self.timeline.library.technique_count - 1, 0),
        )

        layout.addWidget(QLabel("Fret"))
        layout.addWidget(self.fret_input)
        layout.addWidget(QLabel("Technique"))
        layout.addWidget(self.technique_input)
        layout.addWidget(self.shift_label)
        layout.addStretch()
        layout.addWidget(self.create_play_button())
        return controls

    def create_play_button(self) -> QPushButton:
        play_button = QPushButton("Play")
        play_button.clicked.connect(self.timeline.play)
        return play_button

    def refresh_note_items(self) -> None:
        for item in self.note_items:
            self.plots_scene.removeItem(item)
        self.note_items = []

        for event in self.timeline.events:
            item = create_note_item(
                event,
                self.timeline.library.sample(
                    event.technique_index,
                    event.string_index,
                    event.fret,
                ),
                event.id == self.selected_event_id,
                self.on_note_moved,
                self.on_note_selected,
                self.on_note_toggled,
                self.on_note_deleted,
            )
            self.plots_scene.addItem(item)
            self.note_items.append(item)

    def update_scene_rect(self) -> None:
        width = max(
            WINDOW_WIDTH,
            int(self.timeline.length_samples / TIMELINE_SAMPLES_PER_PIXEL) + 100,
        )
        self.plots_scene.setSceneRect(0, 0, width, NUM_STRINGS * PLOT_HEIGHT)

        for item in self.plots_scene.items():
            if item.data(0) == "lane":
                item.setRect(0, item.rect().y(), width, PLOT_HEIGHT)

    def add_note_at(self, string_index: int, start_sample: int) -> None:
        event = self.timeline.add_note(
            string_index=string_index,
            fret=self.fret_input.value(),
            technique_index=self.technique_input.value(),
            start_sample=start_sample,
        )
        if event is None:
            return

        self.update_scene_rect()
        self.refresh_note_items()
        self.analysis_update_timer.start(ANALYSIS_UPDATE_DELAY_MS)

    def on_note_selected(self, event_id: int) -> None:
        self.selected_event_id = event_id
        self.selected_shift_semitones = 0
        self.update_shift_label()
        self.update_note_selection_styles()
        self.update_spectrograms()
        self.setFocus(Qt.FocusReason.MouseFocusReason)

    def update_note_selection_styles(self) -> None:
        for item in self.note_items:
            widget = item.widget()
            if hasattr(widget, "set_selected"):
                widget.set_selected(item.data(1) == self.selected_event_id)

    def on_note_moved(self, event_id: int, x_position: int) -> int:
        event = self.timeline.event_by_id(event_id)
        if event is None:
            return 0

        self.timeline.move_note(
            event_id,
            int(x_position * TIMELINE_SAMPLES_PER_PIXEL),
        )
        self.update_scene_rect()
        self.analysis_update_timer.start(ANALYSIS_UPDATE_DELAY_MS)
        return int(event.start_sample / TIMELINE_SAMPLES_PER_PIXEL)

    def on_note_toggled(self, event_id: int) -> None:
        self.timeline.toggle_note(event_id)
        self.refresh_note_items()
        self.analysis_update_timer.start(ANALYSIS_UPDATE_DELAY_MS)

    def on_note_deleted(self, event_id: int) -> None:
        self.timeline.delete_note(event_id)
        if self.selected_event_id == event_id:
            self.selected_event_id = None
            self.selected_shift_semitones = 0
            self.update_shift_label()
        self.refresh_note_items()
        self.analysis_update_timer.start(ANALYSIS_UPDATE_DELAY_MS)

    def keyPressEvent(self, event) -> None:
        if self.selected_event_id is not None and event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
        ):
            direction = -1 if event.key() == Qt.Key.Key_Left else 1
            self.set_selected_shift(self.selected_shift_semitones + direction)
            event.accept()
            return

        super().keyPressEvent(event)

    def set_selected_shift(self, semitones: int) -> None:
        clamped = min(
            max(semitones, CQT_SHIFT_MIN_SEMITONES),
            CQT_SHIFT_MAX_SEMITONES,
        )
        if clamped == self.selected_shift_semitones:
            return

        self.selected_shift_semitones = clamped
        self.update_shift_label()
        self.update_spectrograms()

    def update_shift_label(self) -> None:
        if self.selected_event_id is None:
            self.shift_label.setText("CQT shift: none")
        else:
            self.shift_label.setText(
                f"CQT shift: {self.selected_shift_semitones:+d} st"
            )

    def update_spectrograms(self) -> None:
        audio = self.timeline.render()

        if AnalysisPlot.STFT in self.spectrogram_plots:
            stft_db, stft_times, stft_freqs = compute_stft(audio)
            self.spectrogram_plots[AnalysisPlot.STFT].update_spectrogram(
                stft_db,
                stft_times[-1] if len(stft_times) else 1,
                stft_freqs[-1] if len(stft_freqs) else 1,
            )

        if AnalysisPlot.CQT in self.spectrogram_plots:
            cqt_db, cqt_times = compute_cqt(audio)
            cqt_plot = self.spectrogram_plots[AnalysisPlot.CQT]
            cqt_plot.update_spectrogram(
                cqt_db,
                cqt_times[-1] if len(cqt_times) else 1,
                cqt_db.shape[0] if cqt_db.ndim else 1,
            )
            self.update_cqt_overlay(cqt_plot, cqt_db.shape[0] if cqt_db.ndim else 1)

        if AnalysisPlot.LOG_BINS in self.spectrogram_plots:
            log_bin_db, log_bin_times, log_bin_centers = compute_log_bins(audio)
            self.spectrogram_plots[AnalysisPlot.LOG_BINS].update_spectrogram(
                log_bin_db,
                log_bin_times[-1] if len(log_bin_times) else 1,
                len(log_bin_centers),
            )

    def update_cqt_overlay(self, plot: SpectrogramPlot, y_max: int) -> None:
        if self.selected_event_id is None:
            plot.clear_overlay()
            return

        event = self.timeline.event_by_id(self.selected_event_id)
        if event is None:
            self.selected_event_id = None
            self.selected_shift_semitones = 0
            self.update_shift_label()
            plot.clear_overlay()
            return

        sample = self.timeline.library.sample(
            event.technique_index,
            event.string_index,
            event.fret,
        )
        visible_bins = int(y_max)
        extra_high_bins = max(-self.selected_shift_semitones, 0)
        source_bins = min(
            visible_bins + extra_high_bins,
            max_cqt_bins(sample_rate=SAMPLE_RATE),
        )
        sample_cqt_db, _ = compute_cqt(sample, n_bins=source_bins)
        overlay_floor = float(np.min(sample_cqt_db)) if sample_cqt_db.size else 0.0
        shifted_cqt_db = shift_frequency_bins(
            sample_cqt_db,
            self.selected_shift_semitones,
            fill_value=overlay_floor,
        )[:visible_bins]
        plot.update_overlay(
            shifted_cqt_db,
            event.start_sample / SAMPLE_RATE,
            event.duration_samples / SAMPLE_RATE,
            y_max,
        )
