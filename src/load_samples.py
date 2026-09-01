"""
    * Each leaf directory contains a specific guitar playing technique only, with 150 one-shot samples, one for each fret.
    * The wav files' name format should be 2 digit zero padded one-indexed pairs, e.g. 01_01 lowest string, open ("fret 0").
"""


import numpy as np
from pathlib import Path
import librosa
import sounddevice as sd


FRETS_PER_STRING = 25
NUM_STRINGS = 6


def normalize_amplitude(audio_array):
    peak = np.max(np.abs(audio_array))
    return audio_array if peak == 0 else audio_array / peak


def load_wav(path, normalize=True):
    data, sr = librosa.load(path, sr=22050, mono=True)
    return (normalize_amplitude(data) if normalize else data), sr


def load_technique_samples(technique_path: Path):
    sorted_wav_paths = sorted(technique_path.glob("*.wav"))
    assert len(sorted_wav_paths) == NUM_STRINGS * FRETS_PER_STRING, (
        f"{technique_path} has {len(sorted_wav_paths)} files, "
        f"expected {NUM_STRINGS * FRETS_PER_STRING}")

    samples = []
    for start in range(0, NUM_STRINGS * FRETS_PER_STRING, FRETS_PER_STRING):
        string_wav_paths = sorted_wav_paths[start:start + FRETS_PER_STRING]
        samples.append([load_wav(p)[0] for p in string_wav_paths])
    return samples


FretSamples = np.ndarray
StringSamples = list[FretSamples]
TechniqueSamples = list[StringSamples]


def load_samples_from_files() -> list[TechniqueSamples]:
    techniques_paths = filter(Path.is_dir, Path("samples/raw").glob("*/*"))

    samples = []
    for technique_idx, technique_path in enumerate(techniques_paths):
        technique_label = " ".join(technique_path.parts[-2:])
        print(f"{technique_idx} : {technique_label}")
        samples.append(load_technique_samples(technique_path))

    return samples


samples = load_samples_from_files()
sd.play(samples[0][0][0])
sd.wait()
