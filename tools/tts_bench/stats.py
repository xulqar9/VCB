"""Dependency-free statistics for the TTS benchmark (mean / median / p95)."""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """q-th percentile (0-100) by linear interpolation between closest ranks:
    the same definition as numpy.percentile's default. NaN for no data."""
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def describe(values: Iterable[float | None]) -> dict:
    """n / mean / median / p95 of the usable values (None and NaN are skipped).
    With nothing usable every statistic is None, so reports can print "-"."""
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p95": percentile(xs, 95),
    }
