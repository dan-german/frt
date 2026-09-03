import sys

from PyQt6.QtWidgets import QApplication

from frt.audio.sample_builder import SampleBuilder
from frt.config import DEFAULT_FRETTING
from frt.ui.main_window import MainWindow


def create_default_sample():
    builder = SampleBuilder()
    return builder.build(DEFAULT_FRETTING)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    window = MainWindow(create_default_sample())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
