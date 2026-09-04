import sys
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QApplication

from frt.audio.sample_library import SampleLibrary
from frt.audio.timeline import Timeline
from frt.ui.main_window import MainWindow

if TYPE_CHECKING:
    from frt.dataset.synthetic import SyntheticExample


def run_timeline(timeline: Timeline, argv: list[str] | None = None) -> int:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if argv is None else argv)

    window = MainWindow(timeline)
    window.show()
    return app.exec()


def run_example(example: "SyntheticExample", argv: list[str] | None = None) -> int:
    if example.timeline is None:
        raise ValueError("example must include a timeline to open it in the app")
    return run_timeline(example.timeline, argv=argv)


def main(argv: list[str] | None = None) -> int:
    return run_timeline(Timeline(SampleLibrary.load()), argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
