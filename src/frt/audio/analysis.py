import numpy as np
import librosa

from frt.config import (
    LOG_BANDS_PER_OCTAVE,
    LOG_BIN_MAX_FREQ,
    LOG_BIN_MIN_FREQ,
    LOG_BIN_REF_FREQ,
    SAMPLE_RATE,
)


def shift_frequency_bins(
    matrix: np.ndarray,
    bins_offset: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    if matrix.ndim == 0:
        raise ValueError("matrix must have at least one axis")

    shifted = np.full(matrix.shape, fill_value, dtype=matrix.dtype)
    row_count = matrix.shape[0]
    if row_count == 0:
        return shifted

    if bins_offset == 0:
        return matrix.copy()

    if abs(bins_offset) >= row_count:
        return shifted

    if bins_offset > 0:
        shifted[bins_offset:] = matrix[: row_count - bins_offset]
    else:
        shifted[: row_count + bins_offset] = matrix[-bins_offset:]

    return shifted


def max_cqt_bins(
    sample_rate: int = SAMPLE_RATE,
    bins_per_octave: int = 12,
    fmin: float | None = None,
) -> int:
    min_frequency = librosa.note_to_hz("C1") if fmin is None else fmin
    nyquist = sample_rate / 2
    return int(np.floor(bins_per_octave * np.log2(nyquist / min_frequency))) + 1


def compute_cqt(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = 256,
    bins_per_octave: int = 12,
    n_bins: int = 108,
    fmin: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cqt = librosa.cqt(
        audio,
        sr=sample_rate,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        n_bins=n_bins,
        fmin=fmin,
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


def make_log_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 2048,
    bands_per_octave: int = LOG_BANDS_PER_OCTAVE,
    min_freq: float = LOG_BIN_MIN_FREQ,
    max_freq: float = LOG_BIN_MAX_FREQ,
    ref_freq: float = LOG_BIN_REF_FREQ,
) -> tuple[np.ndarray, np.ndarray]:
    fft_freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    max_freq = min(max_freq, sample_rate / 2)

    k_min = np.ceil(bands_per_octave * np.log2(min_freq / ref_freq)).astype(int)
    k_max = np.floor(bands_per_octave * np.log2(max_freq / ref_freq)).astype(int)
    k = np.arange(k_min - 1, k_max + 2)
    log_freqs = ref_freq * 2 ** (k / bands_per_octave)

    bins = np.round(log_freqs * n_fft / sample_rate).astype(int)
    bins = np.clip(bins, 0, len(fft_freqs) - 1)
    bins = np.unique(bins)

    filters = []
    center_freqs = []
    for left, center, right in zip(bins[:-2], bins[1:-1], bins[2:]):
        filt = np.zeros(len(fft_freqs))
        filt[left : center + 1] = np.linspace(0, 1, center - left + 1)
        filt[center : right + 1] = np.linspace(1, 0, right - center + 1)

        if filt.sum() != 0:
            filt /= filt.sum()

        filters.append(filt)
        center_freqs.append(fft_freqs[center])

    return np.array(filters), np.array(center_freqs)


def compute_log_bins(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 2048,
    hop_length: int = 512,
    bands_per_octave: int = LOG_BANDS_PER_OCTAVE,
    min_freq: float = LOG_BIN_MIN_FREQ,
    max_freq: float = LOG_BIN_MAX_FREQ,
    ref_freq: float = LOG_BIN_REF_FREQ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann")
    magnitude = np.abs(stft)
    filters, center_freqs = make_log_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        bands_per_octave=bands_per_octave,
        min_freq=min_freq,
        max_freq=max_freq,
        ref_freq=ref_freq,
    )
    log_magnitude = filters @ magnitude
    db = librosa.amplitude_to_db(log_magnitude, ref=np.max)
    times = librosa.times_like(stft, sr=sample_rate, hop_length=hop_length)
    return db, times, center_freqs
