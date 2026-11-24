"""R8-D/E on SYNTHETIC inputs: subset-plan rules; exact probe optimum (vs sklearn at tight tolerance) and copy
weighting; planted identity effect recovered by the probe with a null placebo; analysis runs; sequential CRN trainer is
exact A/A on GPU for both models (skipped without CUDA)."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from evaluation import r8_de_analysis as de_analysis
from experiments.pmis_r8 import de_plan
from models import probe  # noqa: E402


def toy_c6_plan():
    groups = []
    for s in range(3):
        for f in range(4):
            pairs = [{"k": 100 * s + 10 * f + j, "a": j, "b": j + 10, "c": 50 + j, "e": 60 + j, "label": 1 + (j + f) % 2,
                      "sign": 1, "pos_dist": 1, "seed": 7, "session": 1} for j in range(5)]
            if s == 0:
                for p in pairs:
                    p["label"] = 2  # force the fallback rule
            groups.append({"subject": s, "fold": f, "group": 0, "group_seed": 3, "pairs": pairs, "arms": [], "train": [1], "val": [2],
                           "T": [0], "unexposed": [False]})
    return {"groups": groups}


class Plan(unittest.TestCase):
    def test_subset_rules(self):
        sub = de_plan.subset_plan(toy_c6_plan())
        self.assertEqual(sub["n_pairs"], 12)
        self.assertEqual(sub["n_arms"], 60)
        for g in sub["groups"]:
            p = g["pairs"][0]
            want = 1 + g["fold"] % 2
            if g["subject"] > 0:
                self.assertEqual(p["label"], want)
            else:
                self.assertEqual(g["c6_pair_index"], 0)  # fallback: first pair
            self.assertEqual([a["arm"] for a in g["arms"]], ["A", "B", "C", "E", "AA"])
            self.assertEqual(g["arms"][4]["extra"], g["arms"][0]["extra"])
        self.assertEqual(sum(len(de_plan.shard(sub["groups"], k, 5)) for k in range(5)), 12)


class Probe(unittest.TestCase):
    def test_exact_optimum_and_copy_weight(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        F = rng.standard_normal((80, 6))
        y = (F[:, 0] + rng.standard_normal(80) > 0).astype(np.int64)
        rows = np.r_[np.arange(60), 5]  # trial 5 has a copy
        T = np.arange(60, 80)
        p, it, g = probe.fit_score(F, y, rows, T)
        self.assertLess(g, 1e-6)
        u, cnt = np.unique(rows, return_counts=True)
        mu = (cnt[:, None] * F[u]).sum(0) / cnt.sum()
        sd = np.sqrt((cnt[:, None] * (F[u] - mu) ** 2).sum(0) / cnt.sum())
        lr = LogisticRegression(C=1.0, tol=1e-12, max_iter=10000).fit((F[u] - mu) / sd, y[u], sample_weight=cnt)
        q1 = lr.predict_proba((F[T] - mu) / sd)[:, 1]
        self.assertLess(np.max(np.abs(p - np.where(y[T] == 1, q1, 1 - q1))), 1e-6)
        p2, _, _ = probe.fit_score(F, y, np.r_[np.arange(60), 5, 5][:-1], T)  # same multiset -> identical
        self.assertTrue(np.array_equal(p, p2))

    def test_planted_identity_effect(self):
        rng = np.random.default_rng(1)
        n_t, recs = 400, []
        for s in range(6):
            F = rng.standard_normal((n_t, 8)) * 0.5
            y = np.tile([0, 1], n_t // 2)
            F[:, 0] += 0.8 * (2 * y - 1)
            base = np.arange(0, 300)
            T = np.arange(300, 340)
            pairs = []
            for c in (1, 2):
                cand = T[y[T] == c - 1]
                for j in range(0, 8, 2):
                    pairs.append({"k": len(pairs), "a": int(cand[j]), "b": int(cand[j + 1]), "label": c, "pos_dist": 1, "sign": 1,
                                  "c": int(base[y[base] == c - 1][j]), "e": int(base[y[base] == c - 1][j + 1])})
            um = ~np.isin(T, [x for pr in pairs for x in (pr["a"], pr["b"])])
            t = {"subject": s, "fold": 0, "T": T, "base_canon": base, "unexposed": um, "pairs": pairs}
            r, st = probe.run_task(t, F, y)
            self.assertEqual(st["n_fits"], 4 * len(pairs))
            recs += r
        out = de_analysis.analyse_probe(recs, {r["k"]: 0.0 for r in recs}, {}, set())
        self.assertGreater(out["family"]["CBraMod_probe_d"]["mean_points"], 0.0)
        self.assertEqual(out["family"]["CBraMod_probe_d"]["n_negative"], 0)
        self.assertLess(abs(out["descriptive"]["placebo"]["mean_points"]), out["family"]["CBraMod_probe_d"]["mean_points"])


@unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA required")
class GPU(unittest.TestCase):
    def test_seq_group_exact_aa_both_models(self):
        import torch

        from models import models_de, seq_train
        torch.use_deterministic_algorithms(True)
        dev = torch.device("cuda")
        rng = np.random.default_rng(0)
        y = torch.as_tensor(np.tile([0, 1], 50), device=dev)
        g = {"train": list(range(40)), "val": list(range(80, 100)), "T": list(range(40, 60)), "group_seed": 5,
             "pairs": [{"seed": 9}], "arms": [{"pair": 0, "arm": a, "extra": e} for a, e in (("A", 40), ("B", 41), ("C", 3), ("E", 4), ("AA", 40))]}
        ck = ROOT / "data" / "r8_e" / "ckpt" / "cbramod_pretrained_weights.pth"
        for name, X, mk, opt, kw in (
                ("TCFormer", torch.as_tensor(rng.standard_normal((100, 20, 512)).astype(np.float32), device=dev), models_de.make_tcformer, models_de.tcformer_opt, {}),
                ("CBraMod", torch.as_tensor((0.3 * rng.standard_normal((100, 20, 800))).astype(np.float32), device=dev),
                 lambda: models_de.make_cbramod_ft(ck), models_de.cbramod_opt, {"label_smoothing": 0.1, "clip": 1.0})):
            if name == "CBraMod" and not ck.exists():
                continue
            out = seq_train.run_group(mk, opt, g, X, y, dev, 3, 16, **kw)
            self.assertTrue(np.array_equal(out["p_final"][4], out["p_final"][0]), name)
            self.assertTrue(np.array_equal(out["p_es"][4], out["p_es"][0]), name)
            self.assertFalse(np.array_equal(out["p_final"][1], out["p_final"][0]), name)
            self.assertEqual(out["p_final"].shape, (5, 20))


if __name__ == "__main__":
    unittest.main()
