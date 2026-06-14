#!/usr/bin/env python3

"""Backward-compatible wrapper for prepare_physics_inference_input.py."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


if __name__ == "__main__":
    warnings.warn(
        "prepare_two_ball_inference_input.py is deprecated; use prepare_physics_inference_input.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_path(str(Path(__file__).with_name("prepare_physics_inference_input.py")), run_name="__main__")
