import unittest

import numpy as np

from frt.audio.timeline import Timeline, apply_fade_in, apply_fade_out
from frt.config import SAMPLE_RATE


class FakeSampleLibrary:
    technique_count = 1

    def sample(
        self,
        technique_index: int,
        string_index: int,
        fret: int,
    ) -> np.ndarray:
        return np.ones(8, dtype=np.float32)

    def duration_samples(
        self,
        technique_index: int,
        string_index: int,
        fret: int,
    ) -> int:
        return len(self.sample(technique_index, string_index, fret))


class TimelineTest(unittest.TestCase):
    def test_apply_fade_in_tapers_sample_start_from_zero(self) -> None:
        sample = np.ones(8, dtype=np.float32)

        faded = apply_fade_in(sample, fade_in_samples=4)

        np.testing.assert_allclose(
            faded[:4],
            np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(faded[4:], np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(sample, np.ones(8, dtype=np.float32))

    def test_apply_fade_out_tapers_sample_end_to_zero(self) -> None:
        sample = np.ones(8, dtype=np.float32)

        faded = apply_fade_out(sample, fade_out_samples=4)

        np.testing.assert_allclose(faded[:4], np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(
            faded[4:],
            np.array([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(sample, np.ones(8, dtype=np.float32))

    def test_render_applies_fade_out_without_changing_event_duration(self) -> None:
        timeline = Timeline(FakeSampleLibrary(), length_samples=SAMPLE_RATE)
        event = timeline.add_note(string_index=0, fret=0, start_sample=3)

        self.assertIsNotNone(event)
        audio = timeline.render(
            fade_in_seconds=0,
            fade_out_seconds=4 / SAMPLE_RATE,
        )

        self.assertEqual(event.duration_samples, 8)
        np.testing.assert_allclose(audio[3:7], np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(
            audio[7:11],
            np.array([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float32),
        )

    def test_render_applies_fade_in_without_changing_event_duration(self) -> None:
        timeline = Timeline(FakeSampleLibrary(), length_samples=SAMPLE_RATE)
        event = timeline.add_note(string_index=0, fret=0, start_sample=3)

        self.assertIsNotNone(event)
        audio = timeline.render(
            fade_in_seconds=4 / SAMPLE_RATE,
            fade_out_seconds=0,
        )

        self.assertEqual(event.duration_samples, 8)
        np.testing.assert_allclose(
            audio[3:7],
            np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(audio[7:11], np.ones(4, dtype=np.float32))

    def test_render_can_disable_fades(self) -> None:
        timeline = Timeline(FakeSampleLibrary(), length_samples=SAMPLE_RATE)
        timeline.add_note(string_index=0, fret=0, start_sample=0)

        audio = timeline.render(fade_in_seconds=0, fade_out_seconds=0)

        np.testing.assert_allclose(audio[:8], np.ones(8, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
