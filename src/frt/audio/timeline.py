from dataclasses import dataclass

import numpy as np

from frt.audio.playback import play_audio
from frt.audio.sample_library import SampleLibrary
from frt.config import (
    ANALYSIS_TAIL_SECONDS,
    DEFAULT_TIMELINE_SECONDS,
    NUM_STRINGS,
    RENDER_FADE_IN_SECONDS,
    RENDER_FADE_OUT_SECONDS,
    SAMPLE_RATE,
    TIMELINE_GROW_SECONDS,
)


@dataclass
class NoteEvent:
    id: int
    string_index: int
    fret: int
    technique_index: int
    start_sample: int
    duration_samples: int
    enabled: bool = True
    volume: float = 1.0

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.duration_samples


class Timeline:
    def __init__(
        self,
        library: SampleLibrary,
        length_samples: int = DEFAULT_TIMELINE_SECONDS * SAMPLE_RATE,
    ):
        self.library = library
        self.length_samples = length_samples
        self.events: list[NoteEvent] = []
        self._next_event_id = 1

    def add_note(
        self,
        string_index: int,
        fret: int,
        start_sample: int,
        technique_index: int = 0,
        volume: float = 1.0,
    ) -> NoteEvent | None:
        start_sample = max(start_sample, 0)
        duration_samples = self.library.duration_samples(
            technique_index,
            string_index,
            fret,
        )
        if not self.can_place(string_index, start_sample, duration_samples):
            return None

        event = NoteEvent(
            id=self._next_event_id,
            string_index=string_index,
            fret=fret,
            technique_index=technique_index,
            start_sample=start_sample,
            duration_samples=duration_samples,
            volume=volume,
        )
        self._next_event_id += 1
        self.events.append(event)
        self.ensure_length(event.end_sample)
        return event

    def move_note(self, event_id: int, start_sample: int) -> bool:
        event = self.event_by_id(event_id)
        if event is None:
            return False

        start_sample = max(start_sample, 0)
        if not self.can_place(
            event.string_index,
            start_sample,
            event.duration_samples,
            ignore_event_id=event.id,
        ):
            return False

        event.start_sample = start_sample
        self.ensure_length(event.end_sample)
        return True

    def toggle_note(self, event_id: int) -> None:
        event = self.event_by_id(event_id)
        if event is not None:
            event.enabled = not event.enabled

    def delete_note(self, event_id: int) -> None:
        self.events = [event for event in self.events if event.id != event_id]

    def event_by_id(self, event_id: int) -> NoteEvent | None:
        return next((event for event in self.events if event.id == event_id), None)

    def events_for_string(self, string_index: int) -> list[NoteEvent]:
        return sorted(
            (event for event in self.events if event.string_index == string_index),
            key=lambda event: event.start_sample,
        )

    def can_place(
        self,
        string_index: int,
        start_sample: int,
        duration_samples: int,
        ignore_event_id: int | None = None,
    ) -> bool:
        if not 0 <= string_index < NUM_STRINGS:
            return False

        end_sample = start_sample + duration_samples
        for event in self.events_for_string(string_index):
            if event.id == ignore_event_id:
                continue
            if start_sample < event.end_sample and event.start_sample < end_sample:
                return False
        return True

    def ensure_length(self, required_end_sample: int) -> None:
        grow_by = TIMELINE_GROW_SECONDS * SAMPLE_RATE
        while required_end_sample > self.length_samples:
            self.length_samples += grow_by

    def max_event_end_sample(self) -> int:
        return max((event.end_sample for event in self.events), default=0)

    def render_length_samples(self) -> int:
        tail_samples = ANALYSIS_TAIL_SECONDS * SAMPLE_RATE
        return max(self.max_event_end_sample() + tail_samples, SAMPLE_RATE)

    def render(
        self,
        fade_in_seconds: float = RENDER_FADE_IN_SECONDS,
        fade_out_seconds: float = RENDER_FADE_OUT_SECONDS,
    ) -> np.ndarray:
        output = np.zeros(self.render_length_samples(), dtype=np.float32)
        fade_in_samples = seconds_to_samples(fade_in_seconds)
        fade_out_samples = seconds_to_samples(fade_out_seconds)

        for event in self.events:
            if not event.enabled:
                continue

            sample = self.library.sample(
                event.technique_index,
                event.string_index,
                event.fret,
            )
            start = event.start_sample
            end = min(event.end_sample, len(output))
            length = end - start
            if length > 0:
                event_sample = sample[:length]
                event_sample = apply_fade_in(event_sample, fade_in_samples)
                event_sample = apply_fade_out(event_sample, fade_out_samples)
                output[start:end] += (
                    event_sample * event.volume
                )

        return output

    def play(self) -> None:
        play_audio(self.render(), SAMPLE_RATE)


def create_default_timeline(library: SampleLibrary) -> Timeline:
    from frt.config import DEFAULT_FRETTING

    timeline = Timeline(library)
    for string_index, fret in enumerate(DEFAULT_FRETTING):
        if fret is not None:
            timeline.add_note(string_index=string_index,
                              fret=fret, start_sample=0)
    return timeline


def seconds_to_samples(seconds: float) -> int:
    return max(0, int(round(seconds * SAMPLE_RATE)))


def apply_fade_in(sample: np.ndarray, fade_in_samples: int) -> np.ndarray:
    if fade_in_samples <= 0 or len(sample) == 0:
        return sample

    faded = sample.copy()
    fade_length = min(fade_in_samples, len(faded))
    faded[:fade_length] *= np.linspace(
        0.0,
        1.0,
        fade_length,
        dtype=faded.dtype,
    )
    return faded


def apply_fade_out(sample: np.ndarray, fade_out_samples: int) -> np.ndarray:
    if fade_out_samples <= 0 or len(sample) == 0:
        return sample

    faded = sample.copy()
    fade_length = min(fade_out_samples, len(faded))
    faded[-fade_length:] *= np.linspace(
        1.0,
        0.0,
        fade_length,
        dtype=faded.dtype,
    )
    return faded
