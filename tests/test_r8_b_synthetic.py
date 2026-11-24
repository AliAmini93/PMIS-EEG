"""R8-B on SYNTHETIC inputs: arm invariants fail closed; the analysis recovers a planted exposed-only effect and separates
it from a planted support effect; the GPU CRN pair is exact A/A (skipped without CUDA)."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from experiments.pmis_r8 import bench
from evaluation import r8_bench_analysis as bench_analysis  # noqa: E402


def toy_arms():
    T = np.arange(0, 10)
    exposed = np.zeros(40, bool); exposed[:5] = True
    C = np.arange(10, 40)
    return {"T": T, "C": C, "L": np.r_[C, C[:5], T[:5]], "P": np.r_[C, C[:10]], "LCB": C[5:]}, exposed


class Arms(unittest.TestCase):
    def test_valid(self):
        a, e = toy_arms()
        k = bench.check_arms(a, e)
        self.assertEqual(k["k_extra_distinct_L_vs_P"], 5)
        self.assertEqual(k["k_removed_LCB"], 5)

    def test_rejects(self):
        a, e = toy_arms()
        b = dict(a); b["P"] = np.r_[a["C"], a["C"][:9], a["T"][7:8]]
        self.assertRaises(AssertionError, bench.check_arms, b, e)
        b = dict(a); b["L"] = np.r_[a["C"], a["C"][:4], a["T"][:6]]  # an unexposed scored trial copied
        self.assertRaises(AssertionError, bench.check_arms, b, e)
        b = dict(a); b["P"] = a["P"][:-1]
        self.assertRaises(AssertionError, bench.check_arms, b, e)


class Analysis(unittest.TestCase):
    def test_planted(self):
        rng = np.random.default_rng(1)
        n_part, n_per = 14, 200
        subj = np.repeat(np.arange(n_part), n_per)
        y = np.tile([1, 2], n_part * n_per // 2)
        n = subj.size
        seeds = (42, 43, 44)
        A, exposed, tslr = {}, {}, {}
        for s in seeds:
            e = rng.random(n) < 0.5
            base = rng.random(n) < 0.65
            sup = rng.random(n) < 0.02  # support effect: +2 points on every trial
            gain = e & (rng.random(n) < 0.40)  # exposure: flips 40 % of wrong exposed trials (0.35 x 0.40 = +14 BA points)
            cP = (base | sup).astype(float)
            cL = (base | sup | gain).astype(float)
            cC = base.astype(float)
            cLCB = (base & ~(rng.random(n) < 0.02)).astype(float)
            A[s] = {"c_L": cL, "c_P": cP, "c_C": cC, "c_LCB": cLCB,
                    "p_L": 0.5 + 0.1 * cL, "p_P": 0.5 + 0.1 * cP, "p_C": 0.5 + 0.1 * cC, "p_LCB": 0.5 + 0.1 * cLCB}
            exposed[s] = e
            predP = np.where(cP > 0, y, 3 - y)
            tslr[s] = {"r4_pred_L": predP, "r4_pred_P": predP}
        out = bench_analysis.analyse(A, {"subject": subj, "label": y}, exposed, seeds, tslr)
        d = out["descriptive"]
        lp = out["family_holm"]["H1_EEGNet_LEAKED_minus_PLACEBO_BA"]["mean_pp"]
        ex = d["exposed_only_LEAKED_minus_PLACEBO_BA"]["mean_pp"]
        un = d["unexposed_only_LEAKED_minus_PLACEBO_BA"]["mean_pp"]
        self.assertTrue(5.5 < lp < 8.5, lp)  # expected about 0.5 x 14 = 7
        self.assertTrue(11.0 < ex < 17.0, ex)  # expected about 14
        self.assertAlmostEqual(un, 0.0, places=6)
        self.assertGreater(out["family_holm"]["H3_support_bound_below_half_of_identity"]["mean_pp"], 0)
        self.assertAlmostEqual(d["TSLR_C7_same_arms_LEAKED_minus_PLACEBO_BA"]["mean_pp"], 0.0, places=6)


@unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA required")
class GPU(unittest.TestCase):
    def test_crn_pair_exact_aa(self):
        import torch
        torch.use_deterministic_algorithms(True)
        dev = torch.device("cuda")
        rng = np.random.default_rng(0)
        X = torch.as_tensor(rng.standard_normal((60, 11, 512)).astype(np.float32), device=dev)
        y01 = torch.as_tensor(np.tile([0, 1], 30), device=dev)
        a, e = toy_arms()
        out = bench.run_fold({k: v for k, v in a.items()}, X, y01, dev, (1, 2, 3), aa=True)
        self.assertEqual(float(out["aa_max_abs"]), 0.0)
        for k in ("p_L", "p_P", "p_C", "p_LCB"):
            self.assertEqual(out[k].shape, (10,))


if __name__ == "__main__":
    unittest.main()
