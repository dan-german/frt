from pathlib import Path

import numpy as np

from frt.audio import sample_loader


class SampleLibrary:
    def __init__(self, samples: list[sample_loader.TechniqueSamples]):
        self.samples = samples

    @classmethod
    def load(
        cls,
        raw_root: Path = Path("samples/raw"),
        cache_path: Path = Path("samples.npy"),
    ) -> "SampleLibrary":
        return cls(sample_loader.load(raw_root=raw_root, cache_path=cache_path))

    @property
    def technique_count(self) -> int:
        return len(self.samples)

    def sample(
        self,
        technique_index: int,
        string_index: int,
        fret: int,
    ) -> np.ndarray:
        return self.samples[technique_index][string_index][fret]

    def duration_samples(
        self,
        technique_index: int,
        string_index: int,
        fret: int,
    ) -> int:
        return len(self.sample(technique_index, string_index, fret))

