"""
Each leaf directory contains a specific guitar playing technique only, with 150
one-shot samples, one for each fret.

The wav files' name format should be two-digit zero-padded one-indexed pairs,
for example 01_01 for the lowest string, open fret.
"""

from pathlib import Path

import librosa
import numpy as np

from frt.config import FRETS_PER_STRING, NUM_STRINGS, SAMPLE_RATE
from frt.utils import normalize_amplitude

FretSamples = np.ndarray
StringSamples = list[FretSamples]
TechniqueSamples = list[StringSamples]


def load_wav(
    path: Path,
    normalize: bool = True,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    data, sr = librosa.load(path, sr=sample_rate, mono=True)
    return (normalize_amplitude(data) if normalize else data), sr


def load_technique_samples(technique_path: Path) -> TechniqueSamples:
    sorted_wav_paths = sorted(technique_path.glob("*.wav"))
    expected_count = NUM_STRINGS * FRETS_PER_STRING
    if len(sorted_wav_paths) != expected_count:
        raise ValueError(
            f"{technique_path} has {len(sorted_wav_paths)} files, "
            f"expected {expected_count}"
        )

    samples = []
    for start in range(0, expected_count, FRETS_PER_STRING):
        string_wav_paths = sorted_wav_paths[start: start + FRETS_PER_STRING]
        samples.append([load_wav(path)[0] for path in string_wav_paths])
    return samples


def load(
    raw_root: Path = Path("samples/raw"),
    cache_path: Path = Path("samples.npy"),
) -> list[TechniqueSamples]:
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=True).tolist()

    technique_paths = sorted(
        path for path in raw_root.glob("*/*") if path.is_dir())

    samples = []
    for technique_path in technique_paths:
        samples.append(load_technique_samples(technique_path))

    np.save(cache_path, np.array(samples, dtype=object))
    return samples
