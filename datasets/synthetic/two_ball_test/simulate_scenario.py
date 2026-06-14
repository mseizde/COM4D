#!/usr/bin/env python3

"""Backward-compatible wrapper for simulate_physics_scenario.py."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


if __name__ == "__main__":
    warnings.warn(
        "simulate_scenario.py is deprecated; use simulate_physics_scenario.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_path(str(Path(__file__).with_name("simulate_physics_scenario.py")), run_name="__main__")
