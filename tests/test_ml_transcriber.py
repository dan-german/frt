import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from frt.config import FRETS_PER_STRING, NUM_STRINGS, SAMPLE_RATE
from frt.ml.dataset import TranscriptionDataset, collate_transcription_batch
from frt.ml.decoder import decode_predictions
from frt.ml.labels import TargetConfig, labels_to_targets
from frt.ml.model import OnsetsAndFramesModel
from frt.ml.train import compute_losses


class MLTranscriberTest(unittest.TestCase):
    def test_labels_to_targets_places_frame_and_onset_windows(self) -> None:
        frame_targets, onset_targets = labels_to_targets(
            [
                {
                    "start_sample": 1024,
                    "end_sample": 2048,
                    "string_index": 2,
                    "fret": 4,
                    "technique_index": 0,
                    "volume": 1.0,
                }
            ],
            num_frames=8,
            config=TargetConfig(hop_length=512, onset_radius_frames=1),
        )
        # print(frame_targets)

        self.assertEqual(frame_targets.shape,
                         (8, NUM_STRINGS, FRETS_PER_STRING))
        self.assertEqual(onset_targets.shape, frame_targets.shape)
        np.testing.assert_allclose(frame_targets[[2, 3, 4], 2, 4], 1.0)
        np.testing.assert_allclose(onset_targets[[1, 2, 3], 2, 4], 1.0)
        self.assertEqual(float(frame_targets.sum()), 3.0)
        self.assertEqual(float(onset_targets.sum()), 3.0)

    def test_dataset_loader_returns_feature_and_target_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            audio_dir = dataset_dir / "audio"
            annotation_dir = dataset_dir / "annotations"
            audio_dir.mkdir()
            annotation_dir.mkdir()

            duration_samples = 4096
            time = np.arange(duration_samples, dtype=np.float32) / SAMPLE_RATE
            audio = np.sin(2 * np.pi * 220 * time).astype(np.float32)
            sf.write(audio_dir / "example.wav", audio, SAMPLE_RATE)

            annotation = {
                "id": "example",
                "sample_rate": SAMPLE_RATE,
                "duration_samples": duration_samples,
                "labels": [
                    {
                        "start_sample": 512,
                        "end_sample": 2048,
                        "string_index": 0,
                        "fret": 3,
                        "technique_index": 0,
                        "volume": 1.0,
                    }
                ],
            }
            (annotation_dir / "example.json").write_text(
                json.dumps(annotation),
                encoding="utf-8",
            )
            (dataset_dir / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "id": "example",
                        "audio_path": "audio/example.wav",
                        "annotation_path": "annotations/example.json",
                        "sample_rate": SAMPLE_RATE,
                        "duration_samples": duration_samples,
                        "label_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            item = TranscriptionDataset(dataset_dir)[0]

        self.assertEqual(item.example_id, "example")
        self.assertEqual(item.features.ndim, 3)
        self.assertEqual(item.features.shape[0], 1)
        self.assertEqual(item.frame_targets.shape[0], item.features.shape[1])
        self.assertEqual(item.onset_targets.shape, item.frame_targets.shape)
        self.assertGreater(float(item.frame_targets.sum()), 0.0)

    def test_collate_pads_variable_length_examples(self) -> None:
        first = self._fake_item("first", frames=3, frequency_bins=5)
        second = self._fake_item("second", frames=5, frequency_bins=5)

        batch = collate_transcription_batch([first, second])

        self.assertEqual(batch["features"].shape, (2, 1, 5, 5))
        self.assertEqual(batch["frame_targets"].shape,
                         (2, 5, NUM_STRINGS, FRETS_PER_STRING))
        self.assertEqual(batch["mask"].tolist(), [
                         [True, True, True, False, False], [True] * 5])

    def test_model_forward_and_optimizer_step(self) -> None:
        model = OnsetsAndFramesModel(hidden_size=16, conv_channels=8)
        batch = {
            "features": torch.randn(2, 1, 8, 32),
            "frame_targets": torch.zeros(2, 8, NUM_STRINGS, FRETS_PER_STRING),
            "onset_targets": torch.zeros(2, 8, NUM_STRINGS, FRETS_PER_STRING),
            "mask": torch.ones(2, 8, dtype=torch.bool),
        }
        batch["frame_targets"][:, 2, 0, 3] = 1.0
        batch["onset_targets"][:, 2, 0, 3] = 1.0

        outputs = model(batch["features"])
        self.assertEqual(
            outputs["frame_logits"].shape,
            (2, 8, NUM_STRINGS, FRETS_PER_STRING),
        )
        self.assertEqual(outputs["onset_logits"].shape,
                         outputs["frame_logits"].shape)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss, frame_loss, onset_loss = compute_losses(
            outputs,
            batch,
            frame_pos_weight=5.0,
            onset_pos_weight=20.0,
        )
        self.assertGreater(float(frame_loss.detach()), 0.0)
        self.assertGreater(float(onset_loss.detach()), 0.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_decoder_requires_onset_and_frame_to_start_note(self) -> None:
        frame_prob = np.zeros(
            (6, NUM_STRINGS, FRETS_PER_STRING), dtype=np.float32)
        onset_prob = np.zeros_like(frame_prob)
        frame_prob[1:5, 1, 7] = 0.8
        onset_prob[3, 1, 7] = 0.9

        decoded = decode_predictions(
            frame_prob,
            onset_prob,
            hop_length=512,
            from_logits=False,
        )

        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].start_sample, 3 * 512)
        self.assertEqual(decoded[0].end_sample, 5 * 512)
        self.assertEqual(decoded[0].string_index, 1)
        self.assertEqual(decoded[0].fret, 7)

    def _fake_item(self, example_id: str, frames: int, frequency_bins: int):
        from frt.ml.dataset import TranscriptionItem

        return TranscriptionItem(
            example_id=example_id,
            features=torch.ones(1, frames, frequency_bins),
            frame_targets=torch.zeros(frames, NUM_STRINGS, FRETS_PER_STRING),
            onset_targets=torch.zeros(frames, NUM_STRINGS, FRETS_PER_STRING),
        )


if __name__ == "__main__":
    unittest.main()
