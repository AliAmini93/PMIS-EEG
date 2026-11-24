"""Pure paired estimators and participant-cluster inference for PMIS.

Primary scalar is a *probability* contrast; binary correctness (class-balanced = balanced-accuracy
contrast) is secondary. This code never loads EEG and never trains any model.

Aggregation rule (primary and secondary alike): pair score -> mean within (participant, class) ->
equal weight of the two classes -> one value per participant -> participant mean over n participants.
Seeds, folds, sessions and pairs are NOT independent observations; only participants are.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

REQUIRED_LABELS = (1, 2)
MAX_EXACT_N = 20


@dataclass(frozen=True)
class PairedScore:
    """Correct-class probabilities of the two scored trials under the two arm models.
    p_x_given_y = P(correct class of test trial x | model fitted on C + [y'])."""
    participant: str
    label: int
    p_a_given_a: float
    p_a_given_b: float
    p_b_given_a: float
    p_b_given_b: float
    position_distance: Optional[int] = None  # |pos_a - pos_b| in the origin run file (metadata only)

    @property
    def identity_swap_gain(self) -> float:
        """0.5*[(P(a|M_a)-P(a|M_b))+(P(b|M_b)-P(b|M_a))]."""
        for p in (self.p_a_given_a, self.p_a_given_b,
                  self.p_b_given_a, self.p_b_given_b):
            if not (isinstance(p, (int, float)) and isfinite(p) and 0.0 <= p <= 1.0):
                raise ValueError("score must be a finite probability between 0 and 1")
        return 0.5 * ((self.p_a_given_a - self.p_a_given_b)
                      + (self.p_b_given_b - self.p_b_given_a))


@dataclass(frozen=True)
class PairedCorrect:
    """Secondary: correctness indicators (predicted label == true label) under the two arms."""
    participant: str
    label: int
    c_a_given_a: bool
    c_a_given_b: bool
    c_b_given_a: bool
    c_b_given_b: bool
    position_distance: Optional[int] = None

    @property
    def identity_swap_gain(self) -> float:
        for c in (self.c_a_given_a, self.c_a_given_b, self.c_b_given_a, self.c_b_given_b):
            if not isinstance(c, (bool, np.bool_)):
                raise ValueError("correctness indicators must be booleans")
        return 0.5 * ((float(self.c_a_given_a) - float(self.c_a_given_b))
                      + (float(self.c_b_given_b) - float(self.c_b_given_a)))


def correct_class_probability(proba: np.ndarray, classes: Sequence[int], y: Sequence[int]) -> np.ndarray:
    """Pick P(true class) from a (n, n_classes) predict_proba matrix; fails on unknown labels."""
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(classes)
    y = np.asarray(y)
    idx = np.searchsorted(classes, y)
    if proba.ndim != 2 or proba.shape[0] != y.size or proba.shape[1] != classes.size:
        raise ValueError("probability matrix shape mismatch")
    if (idx >= classes.size).any() or not np.array_equal(classes[np.minimum(idx, classes.size - 1)], y):
        raise ValueError("label not among fitted classes")
    if not np.allclose(proba.sum(1), 1.0, atol=1e-9) or (proba < 0).any():
        raise ValueError("rows are not probability vectors")
    return proba[np.arange(y.size), idx]


def participant_class_balanced(scores: Iterable, required_labels=REQUIRED_LABELS,
                               value: Callable = lambda s: s.identity_swap_gain) -> dict[str, float]:
    """Mean within class, then equally weight each class, per participant.

    Pair count does not change participant weight. Seed/fold dependence is NOT treated as extra
    observations; only one participant summary is returned.
    """
    by_person: dict[str, dict[int, list[float]]] = {}
    for s in scores:
        by_person.setdefault(s.participant, {}).setdefault(s.label, []).append(float(value(s)))
    if len(by_person) < 2:
        raise ValueError("Need at least two independent participant groups")
    result: dict[str, float] = {}
    for person, classes in by_person.items():
        if set(classes) != set(required_labels):
            raise ValueError("Each participant must have both required classes")
        result[person] = sum(sum(classes[k]) / len(classes[k]) for k in required_labels) / len(required_labels)
    return result


def participant_scores(scores: Sequence[PairedScore], required_labels=REQUIRED_LABELS) -> dict[str, float]:
    """Primary participant effects (class-balanced mean identity-swap gain in probability units)."""
    return participant_class_balanced(scores, required_labels)


def sign_flip_floor(n: int) -> float:
    """Smallest attainable exact two-sided sign-flip p with n participants (all same sign, no ties)."""
    return 2.0 / (1 << n)


def exact_two_sided_sign_flip(person_scores: Sequence[float]) -> float:
    """Exact 2-sided sign-flip p-value of the mean at PARTICIPANT level (all 2^n patterns).

    Requires symmetry/exchangeability of participant effects under H0 and participants independent
    at inference level. Not suitable for testing each repeated pair or session as independent.
    """
    vals = np.asarray([float(v) for v in person_scores], dtype=np.float64)
    if not (2 <= vals.size <= MAX_EXACT_N) or not np.all(np.isfinite(vals)):
        raise ValueError(f"Expected 2..{MAX_EXACT_N} finite participant scores")
    n = vals.size
    observed = abs(vals.sum())
    tol = 1e-12 * max(1.0, float(np.abs(vals).sum()))
    count = 0
    # enumerate patterns in blocks to bound memory (2^20 x 20 doubles would be 168 MB)
    block_bits = min(n, 14)
    low = ((np.arange(1 << block_bits)[:, None] >> np.arange(block_bits)) & 1) * 2.0 - 1.0
    low_sum = low @ vals[:block_bits]
    for hi in range(1 << (n - block_bits)):
        hi_signs = ((hi >> np.arange(n - block_bits)) & 1) * 2.0 - 1.0
        tot = low_sum + float(hi_signs @ vals[block_bits:])
        count += int(np.count_nonzero(np.abs(tot) >= observed - tol))
    return count / float(1 << n)


def t_interval(vals: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    from scipy.stats import t
    v = np.asarray(vals, dtype=np.float64)
    n = v.size
    if n < 2:
        raise ValueError("need >= 2 values")
    m, se = float(v.mean()), float(v.std(ddof=1)) / sqrt(n)
    q = float(t.ppf(0.5 + level / 2, n - 1))
    return m - q * se, m + q * se


def summarize_values(vals: Sequence[float], scale: float = 100.0) -> dict[str, float | int]:
    """Participant-level summary (scale=100: probability or BA points). No identities exposed."""
    v = np.asarray(vals, dtype=np.float64)
    lo, hi = t_interval(v, 0.95)
    return {"n_independent_participants": int(v.size), "mean_points": float(v.mean()) * scale,
            "ci95_lower_points": lo * scale, "ci95_upper_points": hi * scale,
            "n_positive": int((v > 0).sum()), "exact_sign_flip_p": exact_two_sided_sign_flip(v),
            "sign_flip_floor": sign_flip_floor(int(v.size))}


def summarize_participants(scores: Sequence[PairedScore]) -> dict[str, float | int]:
    """Return effect, t interval and exact sign-flip p without exposing identities (primary)."""
    person = participant_scores(scores)
    vals = tuple(person.values())
    s = summarize_values(vals)
    return {"n_independent_participants": s["n_independent_participants"],
            "mean_probability_points": s["mean_points"],
            "ci95_lower_probability_points": s["ci95_lower_points"],
            "ci95_upper_probability_points": s["ci95_upper_points"],
            "n_positive": s["n_positive"],
            "exact_sign_flip_p": s["exact_sign_flip_p"],
            "sign_flip_floor": s["sign_flip_floor"]}


def summarize_balanced_accuracy_contrast(correct: Sequence[PairedCorrect]) -> dict[str, float | int]:
    """Secondary: participant mean of BA(own-copy arm) - BA(partner-copy arm) over the paired trials."""
    s = summarize_values(tuple(participant_class_balanced(correct).values()))
    return {"n_independent_participants": s["n_independent_participants"],
            "mean_ba_points": s["mean_points"], "ci95_lower_ba_points": s["ci95_lower_points"],
            "ci95_upper_ba_points": s["ci95_upper_points"], "n_positive": s["n_positive"],
            "exact_sign_flip_p": s["exact_sign_flip_p"]}


def position_strata(scores: Sequence, cut: int = 6) -> dict[str, dict]:
    """Pre-declared temporal sensitivity: near (|pos_a-pos_b| <= cut) vs far (> cut) pairs.
    cut = 6 is the G1 metadata median, fixed before any outcome. A participant lacking either class in
    a stratum is dropped from that stratum and counted (as in R4 `summarise`)."""
    out = {}
    for name, keep in (("near", lambda d: d <= cut), ("far", lambda d: d > cut)):
        sub = [s for s in scores if s.position_distance is not None and keep(s.position_distance)]
        by_p: dict[str, set] = {}
        for s in sub:
            by_p.setdefault(s.participant, set()).add(s.label)
        complete = {p for p, labs in by_p.items() if labs == set(REQUIRED_LABELS)}
        all_p = {s.participant for s in scores}
        kept = [s for s in sub if s.participant in complete]
        res = {"n_pairs": len(kept), "n_participants_dropped": len(all_p - complete)}
        if len(complete) >= 2:
            res.update(summarize_values(tuple(participant_class_balanced(kept).values())))
        out[name] = res
    return out


def holm(pvals: dict) -> dict:
    """Holm step-down adjustment (same rule as the frozen R4 r4_stats.holm)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def minimum_detectable_dz(n: int = 14, alpha: float = 0.05, power: float = 0.80) -> float:
    """Standardised participant-level effect (mean / between-participant SD) detectable by a two-sided
    one-sample t test with n participants. Design-only; involves no outcome."""
    from scipy.optimize import brentq
    from scipy.stats import nct, t
    crit = t.ppf(1 - alpha / 2, n - 1)

    def pw(dz):
        nc = dz * sqrt(n)
        lower = float(nct.cdf(-crit, n - 1, nc))
        lower = lower if isfinite(lower) else 0.0   # underflow of a ~0 tail
        return float(nct.sf(crit, n - 1, nc)) + lower - power
    return float(brentq(pw, 1e-6, 3.0))
