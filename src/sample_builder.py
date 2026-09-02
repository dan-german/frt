import sounddevice as sd
from dataclasses import dataclass
import sample_loader
import numpy as np
import pyqtgraph as pg

MAX_SAMPLE_SAMPLES = 4 * 44100


class SampleBatch:
    def __init__(self, samples):
        self.sample_len = min([len(x) for x in samples])
        self.positions = [0 for _ in range(len(samples))]
        self.canvas = np.zeros((len(samples), self.sample_len))
        for index, sample in enumerate(samples):
            self.canvas[index][:self.sample_len] = sample[:self.sample_len]

    def play(self):
        sd.play(self.combine())
        sd.wait()

    def combine(self):
        final = np.zeros(MAX_SAMPLE_SAMPLES)
        for i in range(len(self.canvas)):
            start = self.positions[i]
            final[start:start+self.sample_len] += self.canvas[i]

        return final

    def reposition(self, sample_index: int, new_position: int):
        # print("setting pos", new_position * 40)
        new_position = min(new_position * 40, MAX_SAMPLE_SAMPLES - self.sample_len)
        self.positions[sample_index] = new_position


class SampleBuilder:
    def __init__(self):
        self.samples = sample_loader.load()

    def build(self, fretting):
        all = []
        for guitar_string_index, note in enumerate(fretting):
            if note is None:
                continue
            all.append(self.samples[0][guitar_string_index][note])
        print(f"all len: {len(all)}")
        return SampleBatch(all)


# builder = SampleBuilder()
# sample = builder.build((0, 0, None, None, None, None))
# sample.reposition(0, 11025)
# sample.play()
# 20_000