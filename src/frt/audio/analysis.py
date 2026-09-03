import numpy as np
import librosa

from frt.config import SAMPLE_RATE


def compute_cqt(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = 256,
    bins_per_octave: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    cqt = librosa.cqt(
        audio,
        sr=sample_rate,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    times = librosa.frames_to_time(
        np.arange(cqt_db.shape[1]),
        sr=sample_rate,
        hop_length=hop_length,
    )
    return cqt_db, times


def compute_stft(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann")
    magnitude = np.abs(stft)
    db = librosa.amplitude_to_db(magnitude, ref=np.max)
    times = librosa.times_like(stft, sr=sample_rate, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    return db, times, freqs

