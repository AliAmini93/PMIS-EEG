"""R10-B window geometry, frozen PMIS temporal-response curves and parameter-free prediction rules.

Windows: 2-s windows [o, o + 2 s) after the cue inside Lee2019's 4-s MI period, at 1000 Hz raw offsets o (ms).
Every window is filtered from its own first sample (window-local; the C4/C5 pipeline), so windows of the same trial
share raw samples only through their overlap, never through filter memory. The scored anchor of every trial is
o = 0 ([0, 2) s, C5's scored window). "Lag" = offset of a training window of the same trial relative to the anchor.

Frozen curves R_m(lag) (retained fraction of the exact-copy advantage, correct-class probability scale):
  lags 0-1000 ms: R7-C5 participant-mean fractions d(lag)/d(0) (reports/aggregate/R7_C5_results.json);
  lag 2000 ms (same trial, no shared sample): R6-C4 (TS+LR) / R6-C4b (EEGNet) cohort fraction d(V2.0)/d(V0.0),
  the only frozen value at that lag (a ratio of cohort means, declared).
Interpolation between knots: piecewise linear in lag (frozen). C5's "62.5 ms" copy is int(round(62.5)) = 62 samples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

WLEN_MS = 2000
# raw offset (ms at 1000 Hz) -> (feature source, index in that source's variant axis)
VARIANTS = {0: ("c4", 0), 62: ("c5", 0), 125: ("c5", 1), 250: ("c5", 2), 500: ("c4", 1), 750: ("b", 0),
            1000: ("c4", 2), 1250: ("b", 1), 1500: ("c4", 3), 1750: ("b", 2), 2000: ("c4", 4)}
OFFSETS = tuple(sorted(VARIANTS))                 # 11 offsets; variant index vi = OFFSETS.index(o)
LAGS_B = (0, 62, 125, 250, 500, 1000, 2000)       # Track B single-lag conditions
LAGS_B_NONZERO = LAGS_B[1:]
STRIDES = (1000, 500, 250)                        # Track A sliding-window strides (ms)
C0_STRIDE = 500                                   # conventional contextual benchmark (75 % overlap)
MODELS = ("EEGNet", "TSLR")



def vi(offset: int) -> int:
    return OFFSETS.index(int(offset))


def stride_offsets(stride: int) -> tuple[int, ...]:
    return tuple(range(0, WLEN_MS + 1, int(stride)))


def shared_samples(lag: int) -> int:
    return max(0, WLEN_MS - abs(int(lag)))


def overlap_fraction(lag: float) -> float:
    return max(0.0, 1.0 - abs(float(lag)) / WLEN_MS)


# ------------------------------------------------------------------------------------------------ frozen curves
def frozen_curves(root: Path) -> dict:
    """Read the pre-outcome curve knots from the ported protocol."""
    from experiments.runtime import CONFIG_ROOT, load_protocol

    curves = load_protocol(CONFIG_ROOT / "R10_B_PROTOCOL.json")["frozen_curves"]
    return {model: {int(lag): float(x) for lag, x in knots.items()} for model, knots in curves.items()}




def interp(knots: dict, lag: float) -> float:
    xs = np.array(sorted(knots), dtype=float)
    ys = np.array([knots[k] for k in sorted(knots)], dtype=float)
    lag = float(lag)
    if lag < xs[0] or lag > xs[-1]:
        raise ValueError(f"lag {lag} outside the frozen knots")
    return float(np.interp(lag, xs, ys))


# ------------------------------------------------------------------------------------------------ predictors
def track_b_prediction(knots: dict) -> dict:
    """Track B (one own-trial window per exposed anchor): predicted retention = R(lag). Functional-free."""
    return {lag: interp(knots, lag) for lag in LAGS_B_NONZERO}


def _additive(vals):
    s = float(np.sum(vals))
    return s / (1.0 + s)


def track_a_prediction(knots: dict, stride: int, functional: str = "additive") -> float:
    """Track A: predicted q(s) = E_A(s) / E_X(s), where A adds every non-anchor window of the exposed trial (lags
    s, 2s, ..., 2000) and X adds the same windows plus the exact anchor copy (R = 1).
      additive (PRIMARY, first-order influence: effects of added rows sum): sum R / (1 + sum R)
      nearest (secondary): R(s)          saturating (secondary): 1 - prod(1 - R)"""
    lags = stride_offsets(stride)[1:]
    r = [interp(knots, lag) for lag in lags]
    if functional == "additive":
        return _additive(r)
    if functional == "nearest":
        return r[0]
    if functional == "saturating":
        return float(1.0 - np.prod([1.0 - v for v in r]))
    raise ValueError(functional)


# Baselines (no PMIS curve). Track B: retention per lag. Track A: q per stride, same additive functional.
def baseline_b(name: str, lag: int) -> float:
    if name == "OVL":     # shared-sample fraction (= linear in lag for fixed-length windows)
        return overlap_fraction(lag)
    if name == "BIN":     # any shared sample leaks like an exact copy; none -> no leak
        return 1.0 if shared_samples(lag) > 0 else 0.0
    if name == "EXACT":   # only exact duplicates leak
        return 0.0
    raise ValueError(name)


def baseline_a(name: str, stride: int) -> float:
    lags = stride_offsets(stride)[1:]
    if name == "OVL":     # shared-sample fraction in the primary (additive) functional
        return _additive([overlap_fraction(x) for x in lags])
    if name == "CNT":     # exposure count / density: every own-trial window counts as one exact copy
        return _additive([1.0] * len(lags))
    if name == "BIN":     # overlapping windows count as exact copies, non-overlapping ones as nothing
        return _additive([1.0 if shared_samples(x) > 0 else 0.0 for x in lags])
    if name == "EXACT":
        return 0.0
    raise ValueError(name)


BASELINES_B = ("OVL", "BIN", "EXACT")
BASELINES_A = ("OVL", "CNT", "BIN", "EXACT")
