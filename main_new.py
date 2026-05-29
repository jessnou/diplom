import argparse
import os
import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADAS entrypoint (desktop GUI).")
    parser.add_argument("--desktop", action="store_true", help="Launch desktop GUI (PySide6).")
    args, remaining = parser.parse_known_args(argv)

    root = os.path.dirname(os.path.realpath(__file__))
    sys.argv = [sys.argv[0]] + remaining
    runpy.run_path(os.path.join(root, "main_desktop.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))