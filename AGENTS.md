# Agent Guide

## Project Overview

`frt` is a Python 3.13 package for guitar fret transcription experiments. It includes:

- A PyQt6 desktop app for arranging and previewing note timelines.
- Audio utilities for loading samples, rendering timelines, playback, waveform views, and spectrogram views.
- Synthetic dataset generation for labeled guitar-note audio.
- An onset-and-frames style ML transcriber built with PyTorch.

Primary package code lives in `src/frt/`. Tests live in `tests/`. Raw sample audio lives under `samples/`.

## Environment

This project uses `uv` and `uv_build`.

Common commands:

```bash
uv sync
uv run python -m unittest discover
uv run frt
uv run frt-generate-dataset --help
uv run frt-train-transcriber --help
```

The app depends on local audio/UI capabilities, so headless environments may not be able to run `uv run frt` successfully. Prefer unit tests for non-UI changes.

## Entry Points

Console scripts are declared in `pyproject.toml`:

- `frt` -> `frt.app:main`
- `frt-generate-dataset` -> `frt.dataset.cli:main`
- `frt-train-transcriber` -> `frt.ml.train:main`

`main.py` is also present as a local launcher.

## Code Layout

- `src/frt/config.py`: Shared constants such as sample rate, guitar dimensions, render fades, and UI sizing.
- `src/frt/app.py`: Application startup and timeline window wiring.
- `src/frt/audio/`: Sample loading, analysis, playback, and timeline rendering.
- `src/frt/ui/`: PyQt6/pyqtgraph UI widgets and main window.
- `src/frt/dataset/`: Synthetic riff/example generation and dataset CLI.
- `src/frt/ml/`: Feature extraction, target-label conversion, model, decoding, dataset loading, and training.
- `tests/`: `unittest` coverage for timeline rendering, synthetic datasets, and ML transcription pieces.

## Development Practices

- Keep source imports package-qualified (`from frt...`) as existing code does.
- Keep numeric/audio behavior deterministic where a seed is exposed. Existing dataset tests assert deterministic output.
- Use `numpy.float32` for rendered/generated audio unless a specific API requires otherwise.
- Preserve timeline invariants: notes on the same string must not overlap, event durations come from the sample library, and render fades should not mutate source samples or event duration.
- Keep generated datasets, model checkpoints, and large derived arrays out of git. `.gitignore` already excludes `data/`, `checkpoints/`, and `*.npy`.
- Do not casually rewrite or remove files under `samples/`; they are project inputs, not generated outputs.

## Testing

Run the full test suite with:

```bash
uv run python -m unittest discover
```

For focused runs:

```bash
uv run python -m unittest tests.test_timeline
uv run python -m unittest tests.test_synthetic_dataset
uv run python -m unittest tests.test_ml_transcriber
```

Add or update tests when changing:

- Timeline placement/rendering/fade behavior.
- Synthetic label formats, manifest formats, or sample timing.
- ML feature shapes, target encodings, decoding thresholds, or training loss behavior.

## Style Notes

- The codebase currently uses standard library `unittest`, dataclasses, type hints, and direct functions over heavy abstractions.
- Prefer small, explicit functions that match the current module boundaries.
- Keep comments sparse and useful; add them only for non-obvious audio, timing, or tensor-shape logic.
- Use `Path` for filesystem paths and write text with `encoding="utf-8"`.

## Before Finishing Changes

1. Run the relevant focused tests.
2. Run `uv run python -m unittest discover` when the change touches shared behavior.
3. Check `git status --short` and make sure generated artifacts were not added.
4. Mention any tests that could not be run, especially UI/audio-device checks.
