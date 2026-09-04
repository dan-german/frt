import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from frt.dataset.synthetic import (
    RiffGenerationConfig,
    generate_dataset,
    generate_example,
    write_dataset,
)


class FakeSampleLibrary:
    technique_count = 1

    def sample(
        self,
        technique_index: int,
        string_index: int,
        fret: int,
    ) -> np.ndarray:
        value = (string_index + 1) * 0.01 + (fret + 1) * 0.001
        return np.full(64, value, dtype=np.float32)

    def duration_samples(
        self,
        technique_index: int,
        string_index: int,
        fret: int,
    ) -> int:
        return len(self.sample(technique_index, string_index, fret))


class SyntheticDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.library = FakeSampleLibrary()
        self.config = RiffGenerationConfig(
            duration_seconds=0.25,
            min_events=6,
            max_events=6,
            technique_index=0,
            min_fret=0,
            max_fret=4,
            min_volume=0.5,
            max_volume=0.75,
        )

    def test_generation_is_deterministic_with_seed(self) -> None:
        first = list(
            generate_dataset(
                library=self.library,
                count=2,
                config=self.config,
                seed=123,
            )
        )
        second = list(
            generate_dataset(
                library=self.library,
                count=2,
                config=self.config,
                seed=123,
            )
        )

        self.assertEqual(
            [example.annotation_dict() for example in first],
            [example.annotation_dict() for example in second],
        )
        for left, right in zip(first, second, strict=True):
            np.testing.assert_allclose(left.audio, right.audio)

    def test_labels_match_generated_events_shape(self) -> None:
        example = generate_example(
            library=self.library,
            config=self.config,
            rng=np.random.default_rng(456),
            example_id="example",
        )

        self.assertEqual(len(example.labels), 6)
        for label in example.labels:
            self.assertLess(label.start_sample, label.end_sample)
            self.assertEqual(label.end_sample - label.start_sample, 64)
            self.assertGreaterEqual(label.string_index, 0)
            self.assertLess(label.string_index, 6)
            self.assertGreaterEqual(label.fret, 0)
            self.assertLessEqual(label.fret, 4)
            self.assertEqual(label.technique_index, 0)
            self.assertGreaterEqual(label.volume, 0.5)
            self.assertLessEqual(label.volume, 0.75)

    def test_generated_example_keeps_timeline_for_app_preview(self) -> None:
        example = generate_example(
            library=self.library,
            config=self.config,
            rng=np.random.default_rng(654),
        )

        self.assertIsNotNone(example.timeline)
        self.assertEqual(len(example.timeline.events), len(example.labels))

    def test_generated_notes_do_not_overlap_on_same_string(self) -> None:
        example = generate_example(
            library=self.library,
            config=self.config,
            rng=np.random.default_rng(789),
        )

        for string_index in range(6):
            labels = sorted(
                (
                    label
                    for label in example.labels
                    if label.string_index == string_index
                ),
                key=lambda label: label.start_sample,
            )
            for previous, current in zip(labels, labels[1:], strict=False):
                self.assertLessEqual(previous.end_sample, current.start_sample)

    def test_write_dataset_writes_audio_annotations_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            entries = write_dataset(
                library=self.library,
                output_dir=output_dir,
                count=2,
                config=self.config,
                seed=321,
            )

            self.assertEqual(len(entries), 2)
            for entry in entries:
                audio_path = output_dir / entry["audio_path"]
                annotation_path = output_dir / entry["annotation_path"]
                self.assertTrue(audio_path.exists())
                self.assertTrue(annotation_path.exists())

                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                self.assertEqual(annotation["id"], entry["id"])
                self.assertEqual(len(annotation["labels"]), 6)

            manifest_path = output_dir / "manifest.jsonl"
            manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_lines), 2)


if __name__ == "__main__":
    unittest.main()
