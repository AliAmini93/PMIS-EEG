"""R15-C0 synthetic trial-level simulator for exact-copy contamination, selection and regret.

SYNTHETIC ONLY. No EEG, no model output and no Dreyer2023 R5/R6 value is read or used to set a parameter.

Generative model (per participant i, model m, trial j, training seed k, arm a), on the logit scale:

    logit P(correct) = eta_m + u_i + v_im + w_ims + d_ij + e_imka
      u_i  ~ N(0, su)   participant ability shared by every model
      v_im ~ N(0, sv)   model x participant interaction (sets the between-model correlation su^2/(su^2+sv^2))
      w_ims~ N(0, sw)   model x participant x session interaction (0 when B and C share a session)
      d_ij ~ N(0, sd)   trial difficulty, shared by every model, seed and arm
      e    ~ N(0, se)   training-seed noise of one fit (correlated across arms of a seed by `crn`)

Contamination in an arm that copies a set Q of selection-run (B) trials into training:
  - spill-over on every B trial: logit shift s_m * (|Q| / |B|)  (linear in the number of sibling copies);
  - own-copy memorisation on j in Q: P = 1 - (1 - P_spill) * (1 - r_imj), with
        r_imj = expit(logit(r_m) + N(0, sr)_im + kappa_m * d_ij)   (r_m = 1 -> exact interpolation)
    kappa_m > 0 makes hard trials harder to memorise (the positivity problem of section S3).

Scores are seed-averaged accuracies in percent (balanced classes, so accuracy = balanced accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

TIE_TOL = 0.01          # BA points, as the frozen R12-X rule
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = GH_W / GH_W.sum()


@dataclass(frozen=True)
class Scenario:
    name: str
    roster: tuple
    accB: tuple                 # clean population mean accuracy (%) on B, per model
    accC: tuple = None          # clean population mean accuracy (%) on C (default = accB)
    r: tuple = None             # memorisation (fix) rate of copied trials, per model
    s: tuple = None             # spill-over logit shift at full dose, per model
    kappa: tuple = None         # difficulty dependence of memorisation, per model
    n: int = 83
    TB: int = 40
    TC: int = 40
    K: int = 5
    su: float = 0.5
    sv: float = 0.3
    sw: float = 0.0
    sd: float = 1.0
    se: float = 0.1
    sr: float = 0.0
    crn: float = 0.5            # share of seed noise common to the arms of one seed (common random numbers)
    delta: float = 0.5
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def M(self):
        return len(self.roster)

    def vec(self, name, default):
        v = getattr(self, name)
        return np.full(self.M(), default, float) if v is None else np.asarray(v, float)


def mean_expit(mu, sigma):
    """E[expit(mu + sigma Z)], Z ~ N(0,1), by Gauss-Hermite quadrature."""
    mu = np.asarray(mu, float)
    return np.sum(GH_W * expit(mu[..., None] + sigma * GH_X), axis=-1)


def total_sd(sc: Scenario, session_term=True):
    return float(np.sqrt(sc.su ** 2 + sc.sv ** 2 + (sc.sw ** 2 if session_term else 0) + sc.sd ** 2 + sc.se ** 2))


def eta_for(acc_pct, sigma):
    """Logit intercept whose population mean accuracy equals acc_pct."""
    f = lambda e: mean_expit(e, sigma) - acc_pct / 100.0  # noqa: E731
    return brentq(f, -12, 12)


def intercepts(sc: Scenario):
    sig = total_sd(sc)
    etaB = np.array([eta_for(a, sig) for a in sc.accB])
    accC = sc.accB if sc.accC is None else sc.accC
    etaC = np.array([eta_for(a, sig) for a in accC])
    return etaB, etaC


def population_scores(sc: Scenario, delta: float, n_mc: int = 400_000, seed: int = 0):
    """Population (infinite participants and trials) clean-B, contaminated-B (dose delta, full-B scoring
    population) and C means, by Monte Carlo over the latent terms."""
    rng = np.random.default_rng(seed)
    etaB, etaC = intercepts(sc)
    r, s, kap = sc.vec("r", 0.0), sc.vec("s", 0.0), sc.vec("kappa", 0.0)
    M = sc.M()
    z_common = rng.normal(0, np.sqrt(sc.su ** 2), (n_mc, 1))
    z = z_common + rng.normal(0, sc.sv, (n_mc, M)) + rng.normal(0, sc.sw, (n_mc, M)) + rng.normal(0, sc.se, (n_mc, M))
    d = rng.normal(0, sc.sd, (n_mc, 1))
    lin = etaB + z + d
    clean = expit(lin).mean(0)
    p_sp = expit(lin + s * delta)
    rr = rate(r, rng.normal(0, sc.sr, (n_mc, M)) if sc.sr > 0 else 0.0, kap * d)
    copied = 1 - (1 - p_sp) * (1 - rr)
    contam = delta * copied.mean(0) + (1 - delta) * p_sp.mean(0)
    C = expit(etaC + z + d).mean(0)
    return 100 * clean, 100 * contam, 100 * C


def rate(r, het, diff):
    """Memorisation rate with heterogeneity on the logit scale; r = 1 stays exactly 1, r = 0 stays 0."""
    r = np.asarray(r, float)
    with np.errstate(divide="ignore"):
        base = logit(np.clip(r, 0, 1))
    out = expit(base + het + diff)
    return np.where(r >= 1.0, 1.0, np.where(r <= 0.0, 0.0, out))


def _seed_noise(rng, sc, shape, n_arms):
    common = rng.normal(0, sc.se, shape)
    return [np.sqrt(sc.crn) * common + np.sqrt(1 - sc.crn) * rng.normal(0, sc.se, shape) for _ in range(n_arms)]


def simulate(sc: Scenario, rng, doses=(0.5,), want_trials=False):
    """One synthetic data set.

    Returns dict with
      BBP[delta]: (n, M) clean placebo-arm scores on all of B (row-count matched to that dose; placebo has no effect)
      BBL[delta]: (n, M) contaminated scores on all of B. delta = 0.5 uses the identity swap: the mean of the
                  arm that copies half a and the arm that copies half b, so every B trial is scored once as an own
                  copy and once as an uncopied sibling. Other doses copy a random delta-fraction once.
      BC: (n, M) clean final-refit scores on C.
      trials (optional, delta = 0.5): per-trial seed-mean correctness under own copy and under sibling copy.
    """
    n, M, K, TB, TC = sc.n, sc.M(), sc.K, sc.TB, sc.TC
    etaB, etaC = intercepts(sc)
    r, s, kap = sc.vec("r", 0.0), sc.vec("s", 0.0), sc.vec("kappa", 0.0)
    u = rng.normal(0, sc.su, (n, 1))
    v = rng.normal(0, sc.sv, (n, M))
    wB = rng.normal(0, sc.sw, (n, M)) if sc.sw > 0 else 0.0
    wC = rng.normal(0, sc.sw, (n, M)) if sc.sw > 0 else 0.0
    dB = rng.normal(0, sc.sd, (n, 1, 1, TB))
    dC = rng.normal(0, sc.sd, (n, 1, 1, TC))
    het = rng.normal(0, sc.sr, (n, M, 1, 1)) if sc.sr > 0 else 0.0
    baseB = (etaB + u + v + wB)[:, :, None, None] + dB                 # (n, M, 1, TB)
    baseC = (etaC + u + v + wC)[:, :, None, None] + dC
    rr = rate(r[None, :, None, None], het, kap[None, :, None, None] * dB)  # (n, M, 1, TB)
    n_arms = 2 + 2 * len(doses)
    E = _seed_noise(rng, sc, (n, M, K, 1), n_arms)
    out = {"BBP": {}, "BBL": {}}

    def draw(p):
        return (rng.random(p.shape) < p).astype(np.float32)

    # final clean refit scored on C
    out["BC"] = 100 * draw(expit(baseC + E[0])).mean(axis=(2, 3))
    clean = draw(expit(baseB + E[1]))
    ai = 2
    perm = np.argsort(rng.random((n, TB)), axis=1)
    for delta in doses:
        out["BBP"][delta] = 100 * clean.mean(axis=(2, 3))
        q = int(round(delta * TB))
        if abs(delta - 0.5) < 1e-12:
            arms = []
            masks = []
            for half in (0, 1):
                m = np.zeros((n, TB), bool)
                idx = perm[:, half * (TB // 2):(half + 1) * (TB // 2)]
                np.put_along_axis(m, idx, True, axis=1)
                masks.append(m)
                lin = baseB + E[ai]
                ai += 1
                p_sp = expit(lin + s[None, :, None, None] * delta)
                p = np.where(m[:, None, None, :], 1 - (1 - p_sp) * (1 - rr), p_sp)
                arms.append(draw(p))
            out["BBL"][delta] = 100 * 0.5 * (arms[0].mean(axis=(2, 3)) + arms[1].mean(axis=(2, 3)))
            if want_trials:
                a0, a1 = arms[0], arms[1]                        # (n, M, K, TB)
                ma = masks[0][:, None, None, :]
                own = np.where(ma, a0, a1)                       # own copy: arm that copied the trial
                sib = np.where(ma, a1, a0)                       # same trial, sibling half copied
                out["trials"] = {"own": own, "sib": sib, "clean": clean, "d": dB[:, 0, 0, :]}
        else:
            m = np.zeros((n, TB), bool)
            if q > 0:
                np.put_along_axis(m, perm[:, :q], True, axis=1)
            lin = baseB + E[ai]
            ai += 1
            p_sp = expit(lin + s[None, :, None, None] * delta)
            p = np.where(m[:, None, None, :], 1 - (1 - p_sp) * (1 - rr), p_sp)
            out["BBL"][delta] = 100 * draw(p).mean(axis=(2, 3))
    return out


# ---------------------------------------------------------------------------------------------------------------
# selection rules and regret (E1 = R12-X leave-one-participant-out cross-fitted policy regret)

def sel_rows(S, tol=TIE_TOL):
    """Roster-order tie-break (the frozen R12-X rule), row-wise over the last axis."""
    return np.argmax(S >= S.max(axis=-1, keepdims=True) - tol, axis=-1)


def tie_mask(S, tol=TIE_TOL):
    return S >= S.max(axis=-1, keepdims=True) - tol


def loo_means(X):
    n = X.shape[0]
    return (X.sum(0)[None] - X) / (n - 1)


def e1(BBP, BBL, BC):
    n = BBP.shape[0]
    wc, wl = sel_rows(loo_means(BBP)), sel_rows(loo_means(BBL))
    r = BC[np.arange(n), wc] - BC[np.arange(n), wl]
    return wc, wl, r


def e1_tieaware(BBP, BBL, BC):
    """Expected regret of a policy that breaks ties uniformly at random among models within TIE_TOL."""
    mc, ml = tie_mask(loo_means(BBP)), tie_mask(loo_means(BBL))
    cc = (mc * BC).sum(1) / mc.sum(1)
    cl = (ml * BC).sum(1) / ml.sum(1)
    return mc, ml, cc - cl


def boot_counts(rng, n, n_boot):
    idx = rng.integers(0, n, (n_boot, n))
    Cm = np.zeros((n_boot, n))
    np.add.at(Cm, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    return Cm


def boot_e1(BBP, BBL, BC, Cm, tieaware=False):
    """Vectorised E1 bootstrap with re-selection (leave-one-identity-out), identical to the frozen
    pmis_r12x.analysis.bootstrap(kind='E1') draws for the roster-order rule (asserted in tests)."""
    n = BBP.shape[0]
    den = (n - Cm)[:, :, None]
    den = np.where(den == 0, np.nan, den)
    sp = ((Cm @ BBP)[:, None, :] - Cm[:, :, None] * BBP[None]) / den
    sl = ((Cm @ BBL)[:, None, :] - Cm[:, :, None] * BBL[None]) / den
    if not tieaware:
        wc, wl = sel_rows(sp), sel_rows(sl)
        gap = BC[np.arange(n)[None, :], wc] - BC[np.arange(n)[None, :], wl]
        flip = ((wc != wl) & (Cm > 0)).any(1)
    else:
        mc, ml = tie_mask(sp), tie_mask(sl)
        cc = (mc * BC[None]).sum(2) / mc.sum(2)
        cl = (ml * BC[None]).sum(2) / ml.sum(2)
        gap = cc - cl
        flip = ((mc != ml).any(2) & (Cm > 0)).any(1)
    undefined = (Cm == n).any(1)                 # every draw of one identity: E1 undefined (frozen code -> 0)
    gap = np.nan_to_num(gap)
    R = (Cm * gap).sum(1) / n
    R[undefined] = 0.0
    return R, flip


def classify(any_flip, q05, q975, lower_bound_sesoi=None):
    if not any_flip:
        return "NO_FLIP"
    if q05 > 0:
        return "FLIP_AND_REGRET"
    if q975 < 0:
        return "REVERSE_REGRET"
    return "FLIP_NO_REGRET"


def with_(sc: Scenario, **kw):
    return replace(sc, **kw)
