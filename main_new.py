import argparse
import os
import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADAS entrypoint (desktop GUI or legacy OpenCV runner).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--desktop", action="store_true", help="Запустить desktop GUI (PySide6).")
    mode.add_argument("--legacy", action="store_true", help="Запустить старый OpenCV-режим.")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.realpath(__file__))

    if args.legacy:
        runpy.run_path(os.path.join(root, "main_legacy.py"), run_name="__main__")
        return 0

    # default: desktop
    runpy.run_path(os.path.join(root, "main_desktop.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

