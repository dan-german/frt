from frt.app import run_example, AnalysisPlot

from frt.audio.sample_library import SampleLibrary
from frt.dataset.synthetic import generate_custom_example

library = SampleLibrary.load()

example = generate_custom_example(
    library=library,
    notes=[
        (0, 1),
        (1, 1),
        (2, 1),
    ],
    gap_samples=0,
)

run_example(example, plots=AnalysisPlot.CQT)
