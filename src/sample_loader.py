"""
    * Each leaf directory contains a specific guitar playing technique only, with 150 one-shot samples, one for each fret.
    * The wav files' name format should be 2 digit zero padded one-indexed pairs, e.g. 01_01 lowest string, open ("fret 0").
"""


import numpy as np
from pathlib import Path
import librosa
import sounddevice as sd
import utils


FRETS_PER_STRING = 25
NUM_STRINGS = 6

def load_wav(path, normalize=True):
    data, sr = librosa.load(path, sr=44100, mono=True)
    return (utils.normalize_amplitude(data) if normalize else data), sr


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


def load() -> list[TechniqueSamples]:
    if Path(cache_path_str := "samples.npy").exists():
        return np.load(cache_path_str, allow_pickle=True).tolist()

    techniques_paths = filter(Path.is_dir, Path("samples/raw").glob("*/*"))

    samples = []
    for technique_idx, technique_path in enumerate(techniques_paths):
        technique_label = " ".join(technique_path.parts[-2:])
        print(f"{technique_idx} : {technique_label}")
        samples.append(load_technique_samples(technique_path))
    np.save(cache_path_str, np.array(samples, dtype=object))
    return samples


samples = load()
# sd.play(samples[0][0][0])
# sd.wait()
