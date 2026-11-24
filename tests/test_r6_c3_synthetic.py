"""R6-C3 (Lee2019 replication) code on SYNTHETIC metadata and SYNTHETIC SPD covariances only."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from experiments.pmis_r6 import c3, lee2019
from evaluation import r6_c3_analysis as c3_analysis, r6_mc_stats as mc_stats  # noqa: E402


def fake_meta(n_subj=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subj + 1):
        for r in range(4):
            labs = rng.permutation(np.r_[np.ones(50, int), 2 * np.ones(50, int)])
            for p in range(100):
                rows.append((s, 1 + r // 2, r, p, labs[p]))
    a = np.array(rows)
    return {"subject": a[:, 0], "session": a[:, 1], "run": a[:, 2], "pos": a[:, 3], "label": a[:, 4]}


def fake_covs(label, p=20, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((label.size, p, 2 * p))
    c = a @ np.swapaxes(a, 1, 2) / (2 * p) + 0.1 * np.eye(p)
    c[label == 2] += 0.4 * np.diag(np.linspace(0.2, 1.8, p))
    return c


class Tasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = fake_meta()
        cls.tasks = lee2019.build_tasks(cls.meta)

    def test_structure(self):
        self.assertEqual(len(self.tasks), 12)
        for t in self.tasks:
            self.assertEqual(t["T"].size, 100)
            self.assertEqual(t["base_canon"].size, 300)
            self.assertEqual(len(np.intersect1d(t["T"], t["base_canon"])), 0)
            self.assertEqual(len(t["pairs"]), 40)
            used = [x for p in t["pairs"] for x in (p["a"], p["b"])]
            self.assertEqual(len(used), len(set(used)))
            self.assertEqual(int(t["unexposed"].sum()), 20)
            run = self.meta["run"]
            for p in t["pairs"]:
                self.assertEqual(self.meta["label"][p["a"]], self.meta["label"][p["b"]])
                self.assertEqual(run[p["a"]], run[p["b"]])
                self.assertNotEqual(p["c"], p["e"])
                for z in (p["c"], p["e"]):
                    self.assertIn(z, set(t["base_canon"].tolist()))
                    self.assertEqual(self.meta["session"][z], p["session"])
                    self.assertEqual(self.meta["label"][z], p["label"])
                    self.assertNotEqual(run[z], run[p["a"]])

    def test_deterministic_and_commitment(self):
        t2 = lee2019.build_tasks(self.meta)
        self.assertEqual(c3.tslr_commitment(self.tasks), c3.tslr_commitment(t2))
        t2[0]["pairs"][0] = dict(t2[0]["pairs"][0], sign=-t2[0]["pairs"][0]["sign"])
        self.assertNotEqual(c3.tslr_commitment(self.tasks), c3.tslr_commitment(t2))


class TslrRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = fake_meta(n_subj=2)
        cls.tasks = lee2019.build_tasks(cls.meta)
        c3.G["covs"], c3.G["y"] = fake_covs(cls.meta["label"]), cls.meta["label"].astype(np.int64)
        cls.res = [c3.run_task_tslr(t) for t in cls.tasks[:2]] + [c3.run_task_tslr(t) for t in cls.tasks[4:6]]

    def test_records_finite_and_estimators_agree(self):
        for r in self.res:
            self.assertEqual(len(r["records"]), 40)
            for x in r["records"]:
                for k, v in x.items():
                    self.assertTrue(np.isfinite(v), k)
                for q in ("pAA", "pAB", "pBA", "pBB"):
                    self.assertLess(abs(x[q] - x[f"l_{q}"]), 2e-2)  # sanity bound: lbfgs (tol 1e-4) vs exact optimum
                self.assertGreaterEqual(x["pAA"], x["pAB"] - 0.05)

    def test_analysis_schema(self):
        recs = [dict(x, subject=r["subject"], fold=r["fold"]) for r in self.res for x in r["records"]]
        out = c3_analysis.analyse(recs)
        self.assertEqual(set(out["secondary"]), set(c3_analysis.SECONDARY))
        self.assertIn("met", out["H2_influence_law"])
        self.assertIn("sign_flip_p_monte_carlo", out["primary"])


class EegnetPlan(unittest.TestCase):
    def test_groups(self):
        tasks = lee2019.build_tasks(fake_meta())
        plan = c3.build_eegnet_plan(tasks)
        aa = {}
        for g in plan["groups"]:
            self.assertLessEqual(len(g["arms"]), c3.EEG_M_MAX)
            names = {}
            for a in g["arms"]:
                names.setdefault(a["pair"], set()).add(a["arm"])
            for i, _ in enumerate(g["pairs"]):
                self.assertTrue({"A", "B", "C", "E"} <= names[i])
            aa[(g["subject"], g["fold"])] = aa.get((g["subject"], g["fold"]), 0) + sum(a["arm"] == "AA" for a in g["arms"])
        self.assertTrue(all(v == 1 for v in aa.values()) and len(aa) == len(tasks))
        self.assertEqual(plan["n_pairs"], len(tasks) * 14)
        self.assertEqual(len(c3.eegnet_commitment(plan, tasks)), 64)


class MonteCarlo(unittest.TestCase):
    def test_sign_flip(self):
        p = mc_stats.sign_flip_mc(np.ones(54), draws=200_000)
        self.assertLess(p, 1e-5)
        self.assertGreater(mc_stats.sign_flip_mc(np.random.default_rng(0).normal(0, 1, 54), draws=200_000), 0.01)


if __name__ == "__main__":
    unittest.main()


class C4Synthetic(unittest.TestCase):
    def test_c4_task_and_analysis(self):
        from experiments.pmis_r6 import c4
        from evaluation import r6_c4_analysis as c4_analysis
        meta = fake_meta(n_subj=2)
        n = meta["label"].size
        rng = np.random.default_rng(3)
        base_c = fake_covs(meta["label"], seed=4)
        win = np.zeros((n, 6, 20, 20))
        for w in range(6):  # variants: progressively less similar to the scored window
            noise = fake_covs(meta["label"], seed=10 + w)
            lam = 1.0 - w / 6
            win[:, w] = lam * base_c + (1 - lam) * noise
        c4.N_TRIALS_SAVED = c4.N_TRIALS
        c4.N_TRIALS = n
        try:
            covs, y = c4.stack(win, meta["label"].astype(np.int64))
            c4.G["covs"], c4.G["y"] = covs, y
            tasks = lee2019.build_tasks(meta)
            res = [c4.run_task(t) for t in tasks[:2]] + [c4.run_task(t) for t in tasks[4:6]]
        finally:
            c4.N_TRIALS = c4.N_TRIALS_SAVED
        recs = [dict(x, subject=r["subject"], fold=r["fold"]) for r in res for x in r["records"]]
        out = c4_analysis.analyse(recs)
        self.assertEqual(set(out["primary_curve"]), set(c4.VARIANTS))
        self.assertIn("sign_flip_p_monte_carlo", out["primary_curve"]["V0.0"])
        self.assertGreater(out["primary_curve"]["V0.0"]["mean_points"], 0)
        self.assertTrue(np.isfinite(out["fraction_of_identity_effect_without_shared_samples_d0_over_d100"]))
        # the identity variant's copy equals the scored window exactly
        self.assertAlmostEqual(out["descriptive_mean_tangent_cosine_copy_vs_scored"]["V0.0"], 1.0, places=9)
        del rng


class C3bAnalysis(unittest.TestCase):
    def test_records_and_analysis_on_fake_outputs(self):
        from experiments.pmis_r6 import c2
        from evaluation import r6_c3b_analysis as c3b_analysis
        meta = fake_meta(n_subj=3)
        tasks = lee2019.build_tasks(meta)
        plan = c3.build_eegnet_plan(tasks)
        tmap = {(int(t["subject"]), int(t["fold"])): t for t in tasks}
        rng = np.random.default_rng(0)
        results = {}
        for g in plan["groups"]:
            T = tmap[(g["subject"], g["fold"])]["T"]
            p = rng.uniform(0.2, 0.8, (len(g["arms"]), len(T)))
            for m, a in enumerate(g["arms"]):
                if a["arm"] == "AA":
                    p[m] = p[[i for i, b in enumerate(g["arms"]) if b["pair"] == 0 and b["arm"] == "A"][0]]
            results[(g["subject"], g["fold"], g["group"])] = {
                "p_T": p, "pred_T": rng.integers(0, 2, p.shape).astype(np.int8),
                "train_p_base_mean": np.full(len(g["arms"]), 0.7), "train_acc_base": np.full(len(g["arms"]), 0.8),
                "train_p_extra": np.full(len(g["arms"]), 0.7)}
        recs, aa = c2.pair_records(plan, tmap, results, meta["label"].astype(np.int64))
        self.assertEqual(len(recs), plan["n_pairs"])
        self.assertTrue(all(np.isfinite(r["off_fold"]) for r in recs))  # bystanders exist in every held-out run
        out = c3b_analysis.analyse(recs, {r["k"]: 0.05 for r in recs}, aa)
        self.assertEqual(set(out["secondary"]), set(c3b_analysis.SECONDARY))
        self.assertIn("spearman_rho", out["descriptive"]["H3_participant_corr_d_vs_partner_arm_BA"])
        self.assertEqual(out["descriptive"]["D7_AA_determinism"]["max_abs_probability_difference"], 0.0)


class C4bSynthetic(unittest.TestCase):
    def test_plan_records_analysis(self):
        from experiments.pmis_r6 import c4b
        from evaluation import r6_c4b_analysis as c4b_analysis
        meta = fake_meta(n_subj=3)
        tasks = lee2019.build_tasks(meta)
        plan = c4b.build_plan(tasks)
        aa = {}
        for g in plan["groups"]:
            self.assertLessEqual(len(g["arms"]), c4b.M_MAX)
            for i, pr in enumerate(g["pairs"]):
                names = {a["arm"] for a in g["arms"] if a["pair"] == i}
                self.assertTrue({f"{v}:{s}" for v in c4b.VARIANTS for s in "AB"} <= names)
                for a in g["arms"]:
                    if a["pair"] == i and a["arm"] != "AA":
                        v, s = a["arm"].split(":")
                        self.assertEqual(a["extra"], c4b.VARIANTS.index(v) * c4b.N_TRIALS + (pr["a"] if s == "A" else pr["b"]))
            aa[(g["subject"], g["fold"])] = aa.get((g["subject"], g["fold"]), 0) + sum(a["arm"] == "AA" for a in g["arms"])
        self.assertTrue(all(v == 1 for v in aa.values()) and len(aa) == len(tasks))
        self.assertEqual(plan["n_pairs"], len(tasks) * 10)
        tmap = {(int(t["subject"]), int(t["fold"])): t for t in tasks}
        rng = np.random.default_rng(0)
        results = {}
        for g in plan["groups"]:
            T = tmap[(g["subject"], g["fold"])]["T"]
            p = rng.uniform(0.2, 0.8, (len(g["arms"]), len(T)))
            for m, a in enumerate(g["arms"]):
                if a["arm"] == "AA":
                    p[m] = p[[i for i, b in enumerate(g["arms"]) if b["pair"] == 0 and b["arm"] == "V0.0:A"][0]]
            results[(g["subject"], g["fold"], g["group"])] = {"p_T": p}
        recs, aa_ = c4b.pair_records(plan, tmap, results)
        self.assertEqual(max(aa_), 0.0)
        tslr = {r["k"]: {v: 0.1 for v in c4b.VARIANTS} for r in recs}
        out = c4b_analysis.analyse(recs, tslr, aa_)
        self.assertEqual(set(out["secondary"]), set(c4b_analysis.SECONDARY))
        self.assertIn("computable", out["secondary"]["S5_EEGNet_minus_TSLR_fraction_without_shared_samples"])

    def test_stack_index(self):
        from experiments.pmis_r6 import c4b
        win = np.zeros((c4b.N_TRIALS, 6, 2, 3), dtype=np.float32)
        for w in range(6):
            win[:, w] = w
        st = c4b.stack(win)
        for j, v in enumerate(c4b.VARIANTS):
            self.assertTrue(np.all(st[j * c4b.N_TRIALS:(j + 1) * c4b.N_TRIALS] == c4b.C4_INDEX[v]))
