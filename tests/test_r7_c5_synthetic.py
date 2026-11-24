"""R7-C5 code on SYNTHETIC metadata, covariances and model outputs only."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from test_r6_c3_synthetic import fake_covs, fake_meta  # noqa: E402
from experiments.pmis_r6 import lee2019  # noqa: E402
from experiments.pmis_r7 import c5
from evaluation import r7_c5_analysis as c5_analysis  # noqa: E402


class C5(unittest.TestCase):
    def test_stack(self):
        n = c5.N_TRIALS
        a4 = np.zeros((n, 6, 1, 1), dtype=np.float32)
        a5 = np.zeros((n, 3, 1, 1), dtype=np.float32)
        for i in range(6):
            a4[:, i] = 10 + i
        for i in range(3):
            a5[:, i] = 20 + i
        st = c5.stack(a4, a5)
        expect = [10, 20, 21, 22, 11, 12]
        for j, v in enumerate(expect):
            self.assertTrue(np.all(st[j * n:(j + 1) * n] == v))

    def test_plan_and_tslr_and_analysis(self):
        meta = fake_meta(n_subj=2)
        tasks = lee2019.build_tasks(meta)
        plan = c5.build_eegnet_plan(tasks)
        aa_count = {}
        for g in plan["groups"]:
            self.assertLessEqual(len(g["arms"]), c5.M_MAX)
            aa_count[(g["subject"], g["fold"])] = aa_count.get((g["subject"], g["fold"]), 0) + sum(a["arm"] == "AA" for a in g["arms"])
        self.assertTrue(all(v == 1 for v in aa_count.values()))
        self.assertEqual(plan["n_pairs"], len(tasks) * c5.N_PAIRS)
        n = meta["label"].size
        base = fake_covs(meta["label"], seed=4)
        saved = c5.N_TRIALS
        c5.N_TRIALS = n
        try:
            covs = np.concatenate([(1 - j / 8) * base + (j / 8) * fake_covs(meta["label"], seed=10 + j) for j in range(6)])
            c5.G["covs"], c5.G["y"] = covs, np.tile(meta["label"].astype(np.int64), 6)
            sel = [tasks[0], tasks[4], tasks[5]]
            res = [c5.run_task_tslr(t) for t in sel]
            plan2 = c5.build_eegnet_plan(sel)
        finally:
            c5.N_TRIALS = saved
        trecs = [dict(x, subject=r["subject"], fold=r["fold"]) for r in res for x in r["records"]]
        tmap = {(int(t["subject"]), int(t["fold"])): t for t in sel}
        rng = np.random.default_rng(0)
        results = {}
        for g in plan2["groups"]:
            T = tmap[(g["subject"], g["fold"])]["T"]
            p = rng.uniform(0.3, 0.9, (len(g["arms"]), len(T)))
            for m, a in enumerate(g["arms"]):
                if a["arm"] == "AA":
                    p[m] = p[[i for i, b in enumerate(g["arms"]) if b["pair"] == 0 and b["arm"] == "D0:A"][0]]
            results[(g["subject"], g["fold"], g["group"])] = {"p_T": p}
        erecs, aa = c5.eegnet_pair_records(plan2, tmap, results)
        self.assertEqual(max(aa), 0.0)
        self.assertEqual(len(erecs), len(trecs))
        out = c5_analysis.analyse(trecs, erecs, aa)
        self.assertIn("TSLR", out["curves"])
        self.assertIn("computable", out["fractions"])


if __name__ == "__main__":
    unittest.main()
