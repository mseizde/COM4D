#!/usr/bin/env python3

"""Backward-compatible wrapper for preprocess_physics_outputs.py."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


if __name__ == "__main__":
    warnings.warn(
        "preprocess_two_ball_outputs.py is deprecated; use preprocess_physics_outputs.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_path(str(Path(__file__).with_name("preprocess_physics_outputs.py")), run_name="__main__")
