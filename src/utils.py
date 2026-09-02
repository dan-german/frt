import numpy as np


def normalize_amplitude(audio_array):
    peak = np.max(np.abs(audio_array))
    return audio_array if peak == 0 else audio_array / peak
