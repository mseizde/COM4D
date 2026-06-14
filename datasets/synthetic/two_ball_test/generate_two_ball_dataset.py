#!/usr/bin/env python3

"""Backward-compatible wrapper for generate_physics_dataset.py."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


if __name__ == "__main__":
    warnings.warn(
        "generate_two_ball_dataset.py is deprecated; use generate_physics_dataset.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_path(str(Path(__file__).with_name("generate_physics_dataset.py")), run_name="__main__")
