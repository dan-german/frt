from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from frt.config import FRETS_PER_STRING, NUM_STRINGS
from frt.ml.features import DEFAULT_HOP_LENGTH


@dataclass(frozen=True)
class TargetConfig:
    hop_length: int = DEFAULT_HOP_LENGTH
    onset_radius_frames: int = 1
    num_strings: int = NUM_STRINGS
    frets_per_string: int = FRETS_PER_STRING


def labels_to_targets(
    labels: list[dict[str, Any]],
    num_frames: int,
    config: TargetConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert note annotations into frame and onset targets shaped [T, S, F]."""
    config = config or TargetConfig()
    frame_targets = np.zeros(
        (num_frames, config.num_strings, config.frets_per_string),
        dtype=np.float32,
    )
    onset_targets = np.zeros_like(frame_targets)

    for label in labels:
        string_index = int(label["string_index"])
        fret = int(label["fret"])
        if not 0 <= string_index < config.num_strings:
            continue
        if not 0 <= fret < config.frets_per_string:
            continue

        start_sample = int(label["start_sample"])
        end_sample = int(label["end_sample"])
        start_frame = max(0, start_sample // config.hop_length)
        end_frame = max(
            start_frame,
            math.ceil(end_sample / config.hop_length),
        )
        if start_frame >= num_frames:
            raise ValueError(
                f"label starts outside feature frames: "
                f"start_sample={start_sample}, "
                f"start_frame={start_frame}, "
                f"num_frames={num_frames}"
            )

        frame_end = min(num_frames - 1, end_frame)
        frame_targets[start_frame : frame_end + 1, string_index, fret] = 1.0

        onset_frame = int(round(start_sample / config.hop_length))
        onset_start = max(0, onset_frame - config.onset_radius_frames)
        onset_end = min(num_frames - 1, onset_frame + config.onset_radius_frames)
        onset_targets[onset_start : onset_end + 1, string_index, fret] = 1.0

    return frame_targets, onset_targets
