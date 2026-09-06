from frt.app import run_example, AnalysisPlot

from frt.audio.sample_library import SampleLibrary
from frt.dataset.synthetic import generate_custom_example
from frt.audio.playback import play_audio

library = SampleLibrary.load()

example = generate_custom_example(
    library=library,
    notes=[
        (0, 24),
        (1, 19),
        (2, 14),
        (3, 9),
        (4, 5),
        (5, 0),
        (5, 12),
        (5, 24)
    ],
    gap_samples=0,
)

# play_audio(example.audio)

run_example(example, plots=AnalysisPlot.CQT)
