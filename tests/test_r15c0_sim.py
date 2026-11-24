"""R15-C0 simulator checks: equivalence of the vectorised E1 bootstrap with the frozen R12-X code, and
population-mean calibration of the generator. Synthetic data only."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from evaluation import r12x_analysis as an  # noqa: E402  (frozen; imported read-only)
from experiments.pmis_r15c0 import sim  # noqa: E402


def test_boot_equals_frozen():
    rng = np.random.default_rng(3)
    for trial in range(4):
        n = 40 + 10 * trial
        BBP = np.round(rng.normal(70, 8, (n, 3)) / 1.25) * 1.25
        BBL = BBP + rng.normal([5, 8, 12], 3, (n, 3))
        BC = BBP + rng.normal(0, 4, (n, 3))
        ref = an.bootstrap(BBP, BBL, BC, "E1", n_boot=200, seed=11 + trial)["dist"]
        Cm = sim.boot_counts(np.random.default_rng(11 + trial), n, 200)
        mine, _ = sim.boot_e1(BBP, BBL, BC, Cm)
        assert np.allclose(ref, mine, atol=1e-10), trial
        wc, wl, r = sim.e1(BBP, BBL, BC)
        ref1 = an.e1(BBP, BBL, BC)
        assert np.isclose(ref1["regret"], r.mean())


def test_population_calibration():
    sc = sim.Scenario("t", ("a", "b"), accB=(70.0, 60.0), n=4000, TB=40, K=2)
    out = sim.simulate(sc, np.random.default_rng(0), doses=(0.5,))
    m = out["BBP"][0.5].mean(0)
    assert np.all(np.abs(m - np.array([70.0, 60.0])) < 0.6), m
    clean, contam, C = sim.population_scores(sc, 0.5)
    assert np.allclose(clean, [70, 60], atol=0.2), clean
    assert np.allclose(contam, clean, atol=0.2)            # r = 0, s = 0: no contamination effect


def test_interpolation_saturates():
    sc = sim.Scenario("t", ("a", "b"), accB=(70.0, 60.0), r=(1.0, 1.0), n=300, TB=40, K=2)
    out = sim.simulate(sc, np.random.default_rng(1), doses=(1.0, 0.5))
    assert np.all(out["BBL"][1.0] == 100.0)
    clean, contam, _ = sim.population_scores(sc, 0.5)
    assert np.allclose(contam, 50 + clean / 2, atol=0.2)   # half copied and memorised, half clean


if __name__ == "__main__":
    test_boot_equals_frozen(); test_population_calibration(); test_interpolation_saturates()
    print("r15c0 simulator tests: 3/3 passed")
