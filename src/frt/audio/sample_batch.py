import numpy as np

from frt.config import (
    DRAG_PIXELS_TO_SAMPLES,
    MAX_SAMPLE_SAMPLES,
    SAMPLE_RATE,
)
from frt.audio.analysis import compute_cqt, compute_stft
from frt.audio.playback import play_audio


class SampleBatch:
    def __init__(
        self,
        samples: list[np.ndarray],
        sample_rate: int = SAMPLE_RATE,
        max_samples: int = MAX_SAMPLE_SAMPLES,
    ):
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.sample_len = min((len(sample) for sample in samples), default=0)
        self.positions = [0 for _ in samples]
        self.canvas = np.zeros((len(samples), self.sample_len))

        for index, sample in enumerate(samples):
            self.canvas[index][: self.sample_len] = sample[: self.sample_len]

    def play(self) -> None:
        play_audio(self.combine(), self.sample_rate)

    def cqt(self) -> tuple[np.ndarray, np.ndarray]:
        return compute_cqt(self.combine(), self.sample_rate)

    def stft(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return compute_stft(self.combine(), self.sample_rate)

    def combine(self) -> np.ndarray:
        final = np.zeros(self.max_samples)
        if self.sample_len == 0:
            return final

        for index, sample in enumerate(self.canvas):
            start = self.positions[index]
            end = min(start + self.sample_len, self.max_samples)
            length = end - start
            if length > 0:
                final[start:end] += sample[:length]

        return final

    def set_position_samples(self, sample_index: int, position: int) -> None:
        if not 0 <= sample_index < len(self.positions):
            return

        max_position = max(self.max_samples - self.sample_len, 0)
        self.positions[sample_index] = min(max(position, 0), max_position)

    def set_position_from_pixels(self, sample_index: int, x_position: int) -> None:
        self.set_position_samples(
            sample_index,
            int(x_position * DRAG_PIXELS_TO_SAMPLES),
        )

    def reposition(self, sample_index: int, new_position: int) -> None:
        self.set_position_from_pixels(sample_index, new_position)

