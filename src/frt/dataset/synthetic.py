from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

from frt.audio.sample_library import SampleLibrary
from frt.audio.timeline import NoteEvent, Timeline
from frt.config import FRETS_PER_STRING, NUM_STRINGS, SAMPLE_RATE


@dataclass(frozen=True)
class RiffGenerationConfig:
    duration_seconds: float = 5.0
    min_events: int = 8
    max_events: int = 24
    technique_index: int = 0
    min_fret: int = 0
    max_fret: int = FRETS_PER_STRING - 1
    min_volume: float = 0.5
    max_volume: float = 1.0
    placement_attempts_per_event: int = 100

    def validate(self, library: SampleLibrary) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.min_events < 0:
            raise ValueError("min_events must be non-negative")
        if self.max_events < self.min_events:
            raise ValueError("max_events must be greater than or equal to min_events")
        if not 0 <= self.technique_index < library.technique_count:
            raise ValueError(
                f"technique_index must be between 0 and {library.technique_count - 1}"
            )
        if not 0 <= self.min_fret <= self.max_fret < FRETS_PER_STRING:
            raise ValueError(
                f"fret range must be within 0 and {FRETS_PER_STRING - 1}"
            )
        if self.min_volume < 0:
            raise ValueError("min_volume must be non-negative")
        if self.max_volume < self.min_volume:
            raise ValueError("max_volume must be greater than or equal to min_volume")
        if self.placement_attempts_per_event <= 0:
            raise ValueError("placement_attempts_per_event must be positive")


@dataclass(frozen=True)
class NoteLabel:
    start_sample: int
    end_sample: int
    string_index: int
    fret: int
    technique_index: int
    volume: float


@dataclass
class SyntheticExample:
    audio: np.ndarray
    labels: list[NoteLabel]
    sample_rate: int = SAMPLE_RATE
    example_id: str | None = None
    timeline: Timeline | None = None

    def annotation_dict(self) -> dict:
        return {
            "id": self.example_id,
            "sample_rate": self.sample_rate,
            "duration_samples": int(len(self.audio)),
            "labels": [asdict(label) for label in self.labels],
        }


def labels_from_events(events: list[NoteEvent]) -> list[NoteLabel]:
    return [
        NoteLabel(
            start_sample=int(event.start_sample),
            end_sample=int(event.end_sample),
            string_index=int(event.string_index),
            fret=int(event.fret),
            technique_index=int(event.technique_index),
            volume=float(event.volume),
        )
        for event in sorted(events, key=lambda item: (item.start_sample, item.string_index))
        if event.enabled
    ]


def generate_custom_example(
    library: SampleLibrary,
    notes: Iterable[tuple[int, int]],
    start_sample: int = 0,
    technique_index: int = 0,
    volume: float = 1.0,
    gap_samples: int = 0,
    example_id: str | None = None,
) -> SyntheticExample:
    """Generate an example from explicit string/fret pairs.

    Notes on different strings share the requested start sample. Repeated notes on
    the same string are placed at the earliest non-overlapping sample for that
    string, preserving the input order.
    """
    if not 0 <= technique_index < library.technique_count:
        raise ValueError(
            f"technique_index must be between 0 and {library.technique_count - 1}"
        )
    if volume < 0:
        raise ValueError("volume must be non-negative")
    if gap_samples < 0:
        raise ValueError("gap_samples must be non-negative")

    timeline = Timeline(library)
    next_start_by_string = [max(start_sample, 0)] * NUM_STRINGS
    for string_index, fret in notes:
        if not 0 <= string_index < NUM_STRINGS:
            raise ValueError(f"string_index must be between 0 and {NUM_STRINGS - 1}")
        if not 0 <= fret < FRETS_PER_STRING:
            raise ValueError(f"fret must be between 0 and {FRETS_PER_STRING - 1}")

        note_start = next_start_by_string[string_index]
        event = timeline.add_note(
            string_index=string_index,
            fret=fret,
            start_sample=note_start,
            technique_index=technique_index,
            volume=volume,
        )
        if event is None:
            raise ValueError(
                f"could not place string={string_index}, fret={fret} "
                f"at sample {note_start}"
            )
        next_start_by_string[string_index] = event.end_sample + gap_samples

    return SyntheticExample(
        audio=timeline.render(),
        labels=labels_from_events(timeline.events),
        sample_rate=SAMPLE_RATE,
        example_id=example_id,
        timeline=timeline,
    )


def generate_example(
    library: SampleLibrary,
    config: RiffGenerationConfig | None = None,
    rng: np.random.Generator | None = None,
    example_id: str | None = None,
) -> SyntheticExample:
    config = config or RiffGenerationConfig()
    config.validate(library)
    rng = rng or np.random.default_rng()

    duration_samples = int(round(config.duration_seconds * SAMPLE_RATE))
    timeline = Timeline(library, length_samples=duration_samples)
    target_event_count = int(rng.integers(config.min_events, config.max_events + 1))
    max_attempts = target_event_count * config.placement_attempts_per_event

    placed_events = 0
    for _ in range(max_attempts):
        if placed_events >= target_event_count:
            break

        string_index = int(rng.integers(0, NUM_STRINGS))
        fret = int(rng.integers(config.min_fret, config.max_fret + 1))
        start_sample = int(rng.integers(0, duration_samples))
        volume = float(rng.uniform(config.min_volume, config.max_volume))

        event = timeline.add_note(
            string_index=string_index,
            fret=fret,
            start_sample=start_sample,
            technique_index=config.technique_index,
            volume=volume,
        )
        if event is not None:
            placed_events += 1

    return SyntheticExample(
        audio=timeline.render(),
        labels=labels_from_events(timeline.events),
        sample_rate=SAMPLE_RATE,
        example_id=example_id,
        timeline=timeline,
    )


def generate_dataset(
    library: SampleLibrary,
    count: int,
    config: RiffGenerationConfig | None = None,
    seed: int | None = None,
) -> Iterator[SyntheticExample]:
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = np.random.default_rng(seed)
    for index in range(count):
        yield generate_example(
            library=library,
            config=config,
            rng=rng,
            example_id=f"example_{index:06d}",
        )


def write_dataset(
    library: SampleLibrary,
    output_dir: Path,
    count: int,
    config: RiffGenerationConfig | None = None,
    seed: int | None = None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    annotation_dir = output_dir / "annotations"
    audio_dir.mkdir(exist_ok=True)
    annotation_dir.mkdir(exist_ok=True)

    manifest_entries = []
    for example in generate_dataset(
        library=library,
        count=count,
        config=config,
        seed=seed,
    ):
        if example.example_id is None:
            raise ValueError("generated examples must have an example_id")

        audio_path = audio_dir / f"{example.example_id}.wav"
        annotation_path = annotation_dir / f"{example.example_id}.json"
        sf.write(audio_path, example.audio, example.sample_rate)

        annotation = example.annotation_dict()
        annotation_path.write_text(
            json.dumps(annotation, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_entry = {
            "id": example.example_id,
            "audio_path": str(audio_path.relative_to(output_dir)),
            "annotation_path": str(annotation_path.relative_to(output_dir)),
            "sample_rate": example.sample_rate,
            "duration_samples": int(len(example.audio)),
            "label_count": len(example.labels),
        }
        manifest_entries.append(manifest_entry)

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in manifest_entries),
        encoding="utf-8",
    )
    return manifest_entries
