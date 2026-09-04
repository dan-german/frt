from frt.audio.sample_library import SampleLibrary
from frt.dataset.synthetic import generate_example
from frt.app import run_example

library = SampleLibrary.load()
a = generate_example(library=library)
run_example(a)
