#!/usr/bin/env python3

"""Backward-compatible wrapper for render_physics_from_json.py."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


if __name__ == "__main__":
    warnings.warn(
        "render_from_json.py is deprecated; use render_physics_from_json.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    runpy.run_path(str(Path(__file__).with_name("render_physics_from_json.py")), run_name="__main__")
