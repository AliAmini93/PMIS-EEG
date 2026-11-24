"""R7-C1b: the line-search Karcher solver converges where the R6 unit-step iteration diverges (synthetic only)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from models.vendor_r4 import r4_models as rm  # noqa: E402
from models.indep_tslr import karcher_mean  # noqa: E402
from models.karcher_ls import IndepTSLR2, karcher_ls, _logs  # noqa: E402


def dispersed(n=400, p=11, seed=0, spread=3.0):
    """SPD matrices with widely spread eigenvalues and orientations (log-normal scales)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        ev = np.exp(rng.normal(0, spread, p))
        out.append((q * ev) @ q.T)
    return np.array(out)


def J(m, c, w):
    w = w / w.sum()
    return np.linalg.norm(np.einsum("n,nij->ij", w, _logs(m, c)), "fro")


class KarcherLS(unittest.TestCase):
    def test_converges_where_unit_step_fails(self):
        # extreme per-matrix conditioning with random orientations (log-eigenvalue SD 4) reproduces the HS-Stroke
        # failure mode of the R6 unit-step iteration (it settles at ||J|| ~ 10 instead of 0)
        c = dispersed(n=150, seed=0, spread=4.0)
        w = np.ones(len(c))
        m_ls, info = karcher_ls(c, w)
        self.assertLessEqual(info["grad_norm"], 1e-8)  # float64 floor for this conditioning
        m_unit, it = karcher_mean(c, w, maxiter=300)
        self.assertGreater(J(m_unit, c, w), 1.0, "fixture must reproduce the unit-step failure mode")
        m_r4 = rm.riemann_mean_w(c, w)
        self.assertLess(J(m_r4, c, w), 1e-8)  # R4's adaptive step also converges here
        self.assertLess(np.linalg.norm(m_ls - m_r4) / np.linalg.norm(m_r4), 1e-6)

    def test_matches_r4_on_ordinary_data_and_weights(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((300, 11, 33))
        c = a @ np.swapaxes(a, 1, 2) / 33 + 0.1 * np.eye(11)
        w = rng.integers(1, 4, 300).astype(float)
        m_ls, info = karcher_ls(c, w)
        self.assertLessEqual(info["grad_norm"], 1e-10)
        m_r4 = rm.riemann_mean_w(c, w)
        self.assertLess(np.linalg.norm(m_ls - m_r4) / np.linalg.norm(m_r4), 1e-8)

    def test_monotone_objective(self):
        c = dispersed(n=150, seed=3, spread=4.0)
        w = np.ones(150)
        prev = np.inf
        for it in (1, 2, 5, 10, 30):
            _, info = karcher_ls(c, w, maxiter=it, tol=0.0)
            self.assertLessEqual(info["objective"], prev * (1 + 1e-9))  # monotone up to float64 resolution of f
            prev = info["objective"]

    def test_full_arm(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((200, 11, 33))
        c = a @ np.swapaxes(a, 1, 2) / 33 + 0.1 * np.eye(11)
        y = np.where(np.arange(200) % 2 == 0, 1, 2)
        c[y == 2] += 0.3 * np.eye(11)
        m = IndepTSLR2().fit(c, y, np.ones(200))
        p = m.p_correct(c[:10], y[:10])
        self.assertTrue(np.all((p > 0) & (p < 1)))
        self.assertLessEqual(m.karcher["grad_norm"], 1e-10)


if __name__ == "__main__":
    unittest.main()


class C1bSynthetic(unittest.TestCase):
    def test_task_and_analysis_schema(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_g3_runner_synthetic as t3
        from experiments.pmis_g2 import g3
        from experiments.pmis_r7 import c1b
        from evaluation import r7_c1b_analysis as c1b_analysis
        inst, dist, plan = t3.build(seeds=(42,))
        covs = t3.synthetic_covs(dist)
        y = dist["label"].astype(np.int64)
        c1b.G["covs"], c1b.G["y"] = covs, y
        g3.G["covs"], g3.G["y"] = covs, y
        tasks = plan["seeds"][42]["tasks"]
        res = [c1b.run_task(t) for t in tasks]
        g = [g3.run_task(t) for t in tasks]
        recs = sorted([x for r in res for x in r["records"]], key=lambda q: q["k"])
        grecs = sorted([dict(x, subject=r["subject"]) for r in g for x in r["records"]], key=lambda q: q["k"])
        stored = {}
        for key in ("pAA", "pAB", "pBA", "pBB", "mAA", "mAB", "mBA", "mBB", "pAG", "pBG", "pAC", "pBC", "pAE", "pBE",
                    "subject", "label", "pos_dist", "sign"):
            stored[f"s42_{key}"] = np.array([float(x.get(key, np.nan)) for x in grecs])
        for s in (43, 44, 45, 46):
            for q in ("pAA", "pAB", "pBA", "pBB", "subject", "label"):
                stored[f"s{s}_{q}"] = stored[f"s42_{q}"]
        mapped = []
        for r in recs:
            m = c1b.to_g3_keys(r)
            k = int(r["k"])
            m.update({q: float(stored[f"s42_{q}"][k]) for q in ("subject", "label", "pos_dist", "sign")})
            mapped.append(m)
        pub = g3.analyse(g)[0]
        out = c1b_analysis.analyse(recs, stored, {"results": pub}, mapped)
        self.assertTrue(out["K1_convergence"]["pass"])
        self.assertTrue(out["K2_per_arm_agreement"]["pass"], out["K2_per_arm_agreement"]["max_abs_diff_probability"])
        self.assertIn("primary", out["K3_g3_public_numbers"]["per_quantity"])
