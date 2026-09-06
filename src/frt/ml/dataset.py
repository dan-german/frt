from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from frt.config import SAMPLE_RATE
from frt.ml.features import DEFAULT_HOP_LENGTH, extract_log_bin_features
from frt.ml.labels import TargetConfig, labels_to_targets


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    audio_path: Path
    annotation_path: Path


@dataclass(frozen=True)
class TranscriptionItem:
    example_id: str
    features: torch.Tensor
    frame_targets: torch.Tensor
    onset_targets: torch.Tensor


def read_manifest(dataset_dir: Path) -> list[ManifestEntry]:
    manifest_path = dataset_dir / "manifest.jsonl"
    entries = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            entry = json.loads(line)
            entries.append(
                ManifestEntry(
                    id=str(entry["id"]),
                    audio_path=dataset_dir / entry["audio_path"],
                    annotation_path=dataset_dir / entry["annotation_path"],
                )
            )
    return entries


class TranscriptionDataset(Dataset[TranscriptionItem]):
    def __init__(
        self,
        dataset_dir: Path,
        hop_length: int = DEFAULT_HOP_LENGTH,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.dataset_dir = dataset_dir
        self.entries = read_manifest(dataset_dir)
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> TranscriptionItem:
        entry = self.entries[index]
        audio, sample_rate = sf.read(entry.audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{entry.audio_path} has sample rate {sample_rate}, "
                f"expected {self.sample_rate}"
            )

        annotation = json.loads(entry.annotation_path.read_text(encoding="utf-8"))
        features = extract_log_bin_features(
            audio,
            sample_rate=sample_rate,
            hop_length=self.hop_length,
        )
        num_frames = features.shape[1]
        frame_targets, onset_targets = labels_to_targets(
            annotation["labels"],
            num_frames=num_frames,
            config=TargetConfig(hop_length=self.hop_length),
        )
        return TranscriptionItem(
            example_id=entry.id,
            features=torch.from_numpy(features),
            frame_targets=torch.from_numpy(frame_targets),
            onset_targets=torch.from_numpy(onset_targets),
        )


def collate_transcription_batch(
    items: list[TranscriptionItem],
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")

    max_frames = max(item.features.shape[1] for item in items)
    frequency_bins = items[0].features.shape[2]
    batch_size = len(items)

    features = torch.zeros(
        batch_size,
        1,
        max_frames,
        frequency_bins,
        dtype=torch.float32,
    )
    frame_targets = torch.zeros(
        batch_size,
        max_frames,
        *items[0].frame_targets.shape[1:],
        dtype=torch.float32,
    )
    onset_targets = torch.zeros_like(frame_targets)
    mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)

    for row, item in enumerate(items):
        frame_count = item.features.shape[1]
        features[row, :, :frame_count, :] = item.features
        frame_targets[row, :frame_count] = item.frame_targets
        onset_targets[row, :frame_count] = item.onset_targets
        mask[row, :frame_count] = True

    return {
        "ids": [item.example_id for item in items],
        "features": features,
        "frame_targets": frame_targets,
        "onset_targets": onset_targets,
        "mask": mask,
    }
