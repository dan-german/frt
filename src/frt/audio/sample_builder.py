from collections.abc import Sequence

from frt.audio.sample_batch import SampleBatch
from frt.audio import sample_loader


class SampleBuilder:
    def __init__(self):
        self.samples = sample_loader.load()

    def build(
        self,
        fretting: Sequence[int | None],
        technique_index: int = 0,
    ) -> SampleBatch:
        selected_samples = []
        for guitar_string_index, note in enumerate(fretting):
            if note is None:
                continue
            selected_samples.append(
                self.samples[technique_index][guitar_string_index][note]
            )
        return SampleBatch(selected_samples)
