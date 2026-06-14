#!/usr/bin/env python3

"""Preprocess generalized ball-event physics outputs into COM4D training data.

This is a thin entrypoint over the generalized synthetic physics preprocessor in
``two_ball_test/preprocess_physics_outputs.py``. That implementation is now
backward-compatible with legacy two-ball metadata and supports generalized
metadata with an ``objects`` dictionary where all ``dynamic: true`` objects are
written as parts.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_BALL_DIR = SCRIPT_DIR.parent / "two_ball_test"
if str(TWO_BALL_DIR) not in sys.path:
    sys.path.insert(0, str(TWO_BALL_DIR))

from preprocess_physics_outputs import main  # noqa: E402


if __name__ == "__main__":
    main()
