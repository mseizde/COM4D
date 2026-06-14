#!/usr/bin/env python3

"""Backward-compatible wrapper for run_physics_comparison.py."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


if __name__ == "__main__":
    warnings.warn(
        "run_two_ball_comparison.py is deprecated; use run_physics_comparison.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_path(str(Path(__file__).with_name("run_physics_comparison.py")), run_name="__main__")
