from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from frt.config import FRETS_PER_STRING, NUM_STRINGS
from frt.ml.features import DEFAULT_HOP_LENGTH


@dataclass(frozen=True)
class DecodedNote:
    start_sample: int
    end_sample: int
    string_index: int
    fret: int
    technique_index: int = 0
    volume: float = 1.0


def _to_probabilities(
    values: torch.Tensor | np.ndarray,
    from_logits: bool,
) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu()
        if values.dtype.is_floating_point:
            array = values.numpy()
        else:
            array = values.float().numpy()
    else:
        array = np.asarray(values, dtype=np.float32)

    if from_logits:
        array = 1.0 / (1.0 + np.exp(-array))
    return array


def decode_predictions(
    frame_predictions: torch.Tensor | np.ndarray,
    onset_predictions: torch.Tensor | np.ndarray,
    hop_length: int = DEFAULT_HOP_LENGTH,
    frame_threshold: float = 0.5,
    onset_threshold: float = 0.5,
    from_logits: bool = True,
) -> list[DecodedNote]:
    frame_prob = _to_probabilities(frame_predictions, from_logits=from_logits)
    onset_prob = _to_probabilities(onset_predictions, from_logits=from_logits)
    if frame_prob.ndim == 4:
        if frame_prob.shape[0] != 1:
            raise ValueError("batched decoding expects batch size 1")
        frame_prob = frame_prob[0]
        onset_prob = onset_prob[0]
    if frame_prob.shape != onset_prob.shape:
        raise ValueError("frame and onset predictions must have the same shape")
    if frame_prob.ndim != 3:
        raise ValueError("predictions must be shaped [T, strings, frets]")

    num_frames, num_strings, frets_per_string = frame_prob.shape
    if num_strings != NUM_STRINGS or frets_per_string != FRETS_PER_STRING:
        raise ValueError(
            f"predictions must use {NUM_STRINGS} strings and "
            f"{FRETS_PER_STRING} frets"
        )

    decoded = []
    active_starts: dict[tuple[int, int], int] = {}
    for frame_index in range(num_frames):
        frame_active = frame_prob[frame_index] >= frame_threshold
        onset_active = onset_prob[frame_index] >= onset_threshold

        for string_index in range(num_strings):
            for fret in range(frets_per_string):
                key = (string_index, fret)
                if key in active_starts:
                    if not frame_active[string_index, fret]:
                        decoded.append(
                            DecodedNote(
                                start_sample=active_starts.pop(key) * hop_length,
                                end_sample=frame_index * hop_length,
                                string_index=string_index,
                                fret=fret,
                            )
                        )
                    continue

                if (
                    frame_active[string_index, fret]
                    and onset_active[string_index, fret]
                ):
                    active_starts[key] = frame_index

    end_sample = num_frames * hop_length
    for (string_index, fret), start_frame in sorted(active_starts.items()):
        decoded.append(
            DecodedNote(
                start_sample=start_frame * hop_length,
                end_sample=end_sample,
                string_index=string_index,
                fret=fret,
            )
        )
    return sorted(
        decoded,
        key=lambda note: (note.start_sample, note.string_index, note.fret),
    )
