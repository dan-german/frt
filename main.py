import sys
from pathlib import Path

try:
    from frt.app import main
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from frt.app import main


if __name__ == "__main__":
    raise SystemExit(main())
