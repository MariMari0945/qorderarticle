"""Helpers shared by the PennyLane and NumPy models (no quantum dependencies)."""
from __future__ import annotations

import numpy as np


def steps_to_converge(hist, frac=0.95, window=20):
    """First step at which the smoothed loss has covered `frac` of its total drop."""
    h = np.asarray(hist, float)
    if len(h) < window + 1:
        return len(h)
    sm = np.convolve(h, np.ones(window) / window, mode="valid")
    target = sm[0] - frac * (sm[0] - sm.min())
    return int(np.argmax(sm <= target) + window - 1)
