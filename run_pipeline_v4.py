#!/usr/bin/env python3

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "run_pipeline.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
