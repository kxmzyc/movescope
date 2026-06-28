"""Optional MotionBERT import path setup."""

from __future__ import annotations

import os
import sys


MOTIONBERT_DIR = os.path.join(os.path.dirname(__file__), "MotionBERT")

if os.path.isdir(MOTIONBERT_DIR) and MOTIONBERT_DIR not in sys.path:
    sys.path.insert(0, MOTIONBERT_DIR)
