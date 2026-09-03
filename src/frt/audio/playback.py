import numpy as np
import sounddevice as sd

from frt.config import SAMPLE_RATE


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

