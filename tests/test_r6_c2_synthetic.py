"""Grouped EEGNet (R6-C2) equals separately trained R4 EEGNet-8,2 models; CRN properties. Synthetic data only."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
from models.vendor_r4.r4_models import make_eegnet  # noqa: E402
from models import eegnet_group as eg  # noqa: E402

C, T = 11, 512


def data(n=40, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, C, T, generator=g, dtype=dtype)
    y = torch.arange(n) % 2
    X[y == 1, :3] *= 1.5
    return X, y


def singles(seeds, dtype=torch.float64, p=0.25):
    out = []
    for s in seeds:
        torch.manual_seed(s)
        out.append(make_eegnet(C, T, kern=eg.KERN, p=p).to(dtype))
    return out


class Equivalence(unittest.TestCase):
    def test_eval_forward_equals_singles(self):
        ms = singles([1, 2, 3])
        g = eg.GroupEEGNet(3, C, T).double()
        for m, s in enumerate(ms):
            g.load_single(m, s)
        X, _ = data(30)
        xs = [X[0:10], X[10:20], X[20:30]]
        g.eval()
        out = g(torch.stack(xs, dim=1))
        for m, s in enumerate(ms):
            s.eval()
            torch.testing.assert_close(out[:, m], s(xs[m][:, None]), atol=1e-10, rtol=0)

    def test_training_equals_separate_training_without_dropout(self):
        X, y = data(40)
        rows = np.stack([np.arange(0, 33), np.arange(5, 38), np.arange(7, 40)])
        seeds = [11, 12, 13]
        torch.set_default_dtype(torch.float64)
        try:
            gm = eg.train_group(X, y, rows, np.arange(3), seeds, group_seed=5, device="cpu", epochs=3, bs=8,
                                dropout=False)
            # reference: each model trained alone with the identical schedule
            for m in range(3):
                torch.manual_seed(seeds[m])
                s = make_eegnet(C, T, kern=eg.KERN)  # p=0.25 at construction: R4's init RNG stream
                for mod in s.modules():
                    if isinstance(mod, nn.Dropout):
                        mod.p = 0.0
                opt = torch.optim.Adam(s.parameters(), lr=1e-3)
                gen = torch.Generator().manual_seed(seeds[m] + 7_919)
                s.train()
                for _ in range(3):
                    perm = torch.randperm(rows.shape[1], generator=gen)
                    order = torch.as_tensor(rows[m])[perm]
                    for i in range(0, order.numel(), 8):
                        b = order[i:i + 8]
                        if b.numel() < 2:
                            continue
                        opt.zero_grad(set_to_none=True)
                        nn.CrossEntropyLoss()(s(X[b][:, None]), y[b]).backward()
                        opt.step()
                        s.max_norm()
                s.eval()
                gm.eval()
                xe = X[:12]
                torch.testing.assert_close(gm(xe[:, None].expand(-1, 3, -1, -1))[:, m], s(xe[:, None]),
                                           atol=1e-8, rtol=0)
        finally:
            torch.set_default_dtype(torch.float32)


class FastPathEqualsConvPath(unittest.TestCase):
    def test_forward_backward_equal(self):
        torch.manual_seed(0)
        ms = singles([4, 5])
        g = eg.GroupEEGNet(2, C, T).double()
        for m, s_ in enumerate(ms):
            g.load_single(m, s_)
        X, _ = data(16)
        x = torch.stack([X[:8], X[8:]], dim=1)
        g.train()
        outs, grads = {}, {}
        for impl in ("conv", "fast"):
            g.impl = impl
            g.zero_grad()
            o = g(x)
            (o ** 2).sum().backward()
            outs[impl] = o.detach().clone()
            grads[impl] = {n: p.grad.detach().clone() for n, p in g.named_parameters()}
        torch.testing.assert_close(outs["fast"], outs["conv"], atol=1e-10, rtol=0)
        for n in grads["conv"]:
            torch.testing.assert_close(grads["fast"][n], grads["conv"][n], atol=1e-9, rtol=1e-9, msg=n)
        g.impl = "fast"


class CommonRandomNumbers(unittest.TestCase):
    def test_identical_arms_identical_models(self):
        X, y = data(40, dtype=torch.float32)
        r = np.arange(0, 33)
        rows = np.stack([r, r, r])
        gm = eg.train_group(X, y, rows, np.array([0, 0, 1]), [21, 22], group_seed=9, device="cpu", epochs=2, bs=8)
        p = eg.predict_proba(gm, X, np.arange(40))
        self.assertTrue(torch.equal(p[0], p[1]))          # A/A within a pair: exactly equal
        self.assertFalse(torch.equal(p[0], p[2]))         # different pair seed: different model

    def test_one_row_swap_is_the_only_difference(self):
        X, y = data(40, dtype=torch.float32)
        base = np.arange(0, 32)
        rows = np.stack([np.r_[base, 35], np.r_[base, 36]])
        gm = eg.train_group(X, y, rows, np.array([0, 0]), [31], group_seed=3, device="cpu", epochs=0)
        p = eg.predict_proba(gm, X, np.arange(40))
        self.assertTrue(torch.equal(p[0], p[1]))          # identical initialisation
        gm = eg.train_group(X, y, rows, np.array([0, 0]), [31], group_seed=3, device="cpu", epochs=2, bs=8)
        p = eg.predict_proba(gm, X, np.arange(40))
        self.assertFalse(torch.equal(p[0], p[1]))

    def test_masks_shared_within_pair(self):
        gen = torch.Generator().manual_seed(0)
        m1, m2 = eg.draw_masks(4, 3, gen, "cpu", 128, 16)
        pair_of = torch.tensor([0, 0, 1, 2])
        a = m1[:, pair_of]
        self.assertTrue(torch.equal(a[:, 0], a[:, 1]))
        self.assertFalse(torch.equal(a[:, 1], a[:, 2]))
        self.assertAlmostEqual(float((m1 > 0).float().mean()), 0.75, delta=0.03)

    def test_predict_own_matches_predict_proba(self):
        X, y = data(40, dtype=torch.float32)
        rows = np.stack([np.arange(0, 33), np.arange(3, 36)])
        gm = eg.train_group(X, y, rows, np.array([0, 1]), [1, 2], group_seed=1, device="cpu", epochs=1, bs=8)
        own = eg.predict_own(gm, X, rows)
        full = eg.predict_proba(gm, X, np.arange(40))
        torch.testing.assert_close(own[1], full[1, rows[1]], atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------ C2 plan / records / analysis (synthetic)


class C2Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_g3_runner_synthetic as t3
        from experiments.pmis_r6 import c2
        cls.c2 = c2
        cls.inst, cls.dist, plan3 = t3.build(seeds=(42,))
        cls.tasks = plan3["seeds"][42]["tasks"]
        cls.plan = c2.build_plan(cls.tasks)

    def test_sampling_outcome_blind_and_third(self):
        for t in self.tasks:
            s = self.c2.sample_pairs(t)
            self.assertEqual(s, self.c2.sample_pairs(dict(t)))
            self.assertEqual(len(s), -(-len(t["pairs"]) // 3))
            self.assertEqual(len(set(s)), len(s))

    def test_groups_keep_pairs_whole_and_one_aa_per_task(self):
        seen = set()
        aa = {}
        for g in self.plan["groups"]:
            self.assertLessEqual(len(g["arms"]), self.c2.M_MAX)
            names = {}
            for a in g["arms"]:
                names.setdefault(a["pair"], set()).add(a["arm"])
            for p_idx, pr in enumerate(g["pairs"]):
                self.assertTrue({"A", "B"} <= names[p_idx])
                self.assertEqual("G" in names[p_idx], pr["g"] >= 0)
                self.assertEqual("C" in names[p_idx], pr["c"] >= 0)
                self.assertNotIn(pr["k"], seen)
                seen.add(pr["k"])
            aa[(g["subject"], g["fold"])] = aa.get((g["subject"], g["fold"]), 0) + sum(a["arm"] == "AA" for a in g["arms"])
        self.assertTrue(all(v == 1 for v in aa.values()))
        self.assertEqual(len(aa), len(self.tasks))

    def test_commitment_sensitive(self):
        c = self.c2.plan_commitment(self.plan, self.tasks)
        p2 = {"groups": [dict(g) for g in self.plan["groups"]]}
        p2["groups"][0] = dict(p2["groups"][0], group_seed=p2["groups"][0]["group_seed"] + 1)
        self.assertNotEqual(c, self.c2.plan_commitment(p2, self.tasks))

    def test_records_and_analysis_on_fake_outputs(self):
        from evaluation import r6_c2_analysis as c2_analysis
        y = self.dist["label"].astype(np.int64)
        tmap = {(int(t["subject"]), int(t["fold"])): t for t in self.tasks}
        rng = np.random.default_rng(0)
        results = {}
        for g in self.plan["groups"]:
            T = tmap[(g["subject"], g["fold"])]["T"]
            p = rng.uniform(0.2, 0.8, (len(g["arms"]), len(T)))
            for m, a in enumerate(g["arms"]):
                if a["arm"] == "AA":
                    p[m] = p[[i for i, b in enumerate(g["arms"]) if b["pair"] == 0 and b["arm"] == "A"][0]]
            results[(g["subject"], g["fold"], g["group"])] = {
                "p_T": p, "pred_T": rng.integers(0, 2, p.shape).astype(np.int8),
                "train_p_base_mean": np.full(len(g["arms"]), 0.7), "train_acc_base": np.full(len(g["arms"]), 0.8),
                "train_p_extra": np.full(len(g["arms"]), 0.7)}
        recs, aa = self.c2.pair_records(self.plan, tmap, results, y)
        self.assertEqual(len(recs), self.plan["n_pairs"])
        self.assertEqual(max(aa), 0.0)
        tslr = {r["k"]: {"pAA": 0.6, "pAB": 0.5, "pBA": 0.5, "pBB": 0.6} for r in recs}
        out = c2_analysis.analyse(recs, tslr, aa)
        self.assertEqual(set(out["secondary"]), set(c2_analysis.SECONDARY))
        self.assertIn("mean_points", out["primary"])


@unittest.skipUnless(torch.cuda.is_available(), "GPU smoke test")
class GpuDeterminism(unittest.TestCase):
    def test_aa_identical_on_gpu_fast_path(self):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 2**30:
            self.skipTest("GPU busy")
        torch.use_deterministic_algorithms(True)
        try:
            dev = torch.device("cuda")
            g = torch.Generator().manual_seed(0)
            X = torch.randn(300, C, T, generator=g).to(dev)
            y = (torch.arange(300) % 2).to(dev)
            base = np.arange(0, 250)
            rows = np.stack([np.r_[base, 260], np.r_[base, 261], np.r_[base, 260], np.r_[base, 262]])
            pair_of = np.array([0, 0, 0, 1])
            m = eg.train_group(X, y, rows, pair_of, [5, 6], 3, dev, epochs=2)
            p = eg.predict_proba(m, X, np.arange(250, 300))
            self.assertTrue(torch.equal(p[0], p[2]))
            self.assertFalse(torch.equal(p[0], p[1]))
            m2 = eg.train_group(X, y, rows, pair_of, [5, 6], 3, dev, epochs=2)
            self.assertTrue(torch.equal(eg.predict_proba(m2, X, np.arange(250, 300)), p))  # run-to-run determinism
        finally:
            torch.use_deterministic_algorithms(False)
