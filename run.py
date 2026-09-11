#!/usr/bin/env python3
"""Border Guard AI launcher.

    python run.py

Checks that the dependencies the application cannot start without are present,
explains clearly what is missing if they are not, and then hands over to
``app.main``.

The check exists because "ModuleNotFoundError: No module named 'PySide6'" is a
worse first experience than a sentence telling you which command to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

MIN_PYTHON = (3, 9)

#: import name -> (pip name, what it is for)
REQUIRED = {
    "PySide6": ("PySide6", "the desktop interface"),
    "cv2": ("opencv-python", "video decoding and image processing"),
    "numpy": ("numpy", "array maths"),
    "yaml": ("PyYAML", "reading config/*.yaml"),
    "torch": ("torch", "the neural network runtime"),
    "ultralytics": ("ultralytics", "YOLO detection and tracking"),
    "sklearn": ("scikit-learn", "behavioural anomaly detection"),
}

#: Nice to have. Absence disables one feature, never the application.
OPTIONAL = {
    "easyocr": ("easyocr", "number plate OCR (better on plates)"),
    "pytesseract": ("pytesseract", "number plate OCR (needs the tesseract binary)"),
    "joblib": ("joblib", "saving the anomaly model between sessions"),
}


def _check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        print(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required.\n"
            f"You are running {sys.version.split()[0]}.\n"
            f"Python 3.11 is recommended.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _check_dependencies() -> None:
    import importlib

    missing: list[tuple[str, str]] = []
    for module, (package, purpose) in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append((package, purpose))

    if missing:
        print("\nBorder Guard AI cannot start: required packages are missing.\n",
              file=sys.stderr)
        for package, purpose in missing:
            print(f"  {package:<18} {purpose}", file=sys.stderr)
        print(
            "\nInstall everything with:\n"
            "    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    absent = []
    for module, (package, purpose) in OPTIONAL.items():
        try:
            importlib.import_module(module)
        except ImportError:
            absent.append((package, purpose))

    # Only mention OCR if neither backend is present.
    ocr_names = {"easyocr", "pytesseract"}
    absent_names = {name for name, _ in absent}
    if ocr_names <= absent_names:
        print(
            "Note: no OCR backend is installed, so number plate recognition "
            "is switched off.\n"
            "      Everything else works. To enable it:\n"
            "          pip install easyocr\n"
            "      or  pip install pytesseract   (plus 'brew install tesseract')\n"
        )


def _check_project_layout() -> None:
    missing = [
        name for name in ("config", "core", "database", "inputs", "app")
        if not (PROJECT_ROOT / name).is_dir()
    ]
    if missing:
        print(
            f"These project folders are missing: {', '.join(missing)}\n"
            f"Run this script from inside the project directory.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    _check_python()
    _check_project_layout()
    _check_dependencies()

    from app.main import main as run_app
    return run_app(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
