import sys

from PyQt6.QtWidgets import QApplication

from frt.audio.sample_library import SampleLibrary
from frt.audio.timeline import Timeline
from frt.ui.main_window import MainWindow

# def create_default_timeline_from_disk():
#     return 


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)

    window = MainWindow(Timeline(SampleLibrary.load()))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
