"""R7-C6 grouped deep decoders (synthetic only): grouped ShallowConvNet equals the sequential reference; CRN."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
from models import deep_group as dg  # noqa: E402

C, T = 20, 512


def data(n=40, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, C, T, generator=g, dtype=dtype)
    y = torch.arange(n) % 2
    X[y == 1, :4] *= 1.5
    return X, y


class Shallow(unittest.TestCase):
    def test_grouped_equals_sequential_forward_backward(self):
        torch.set_default_dtype(torch.float64)
        try:
            singles = []
            for s in (1, 2):
                torch.manual_seed(s)
                singles.append(dg.ShallowSingle(C, T))
            g = dg.GroupShallow(2, C, T)
            for m, s_ in enumerate(singles):
                g.load_single(m, s_)
            X, _ = data(16)
            xs = [X[:8], X[8:]]
            for s_ in singles:
                s_.drop.p = 0.0
                s_.train()
            g.train()
            out = g(torch.stack(xs, 1))
            (out ** 2).sum().backward()
            for m, s_ in enumerate(singles):
                o = s_(xs[m][:, None])
                torch.testing.assert_close(out[:, m], o, atol=1e-10, rtol=0)
                (o ** 2).sum().backward()
                torch.testing.assert_close(g.fc_w.grad[m], s_.fc.weight.grad, atol=1e-8, rtol=1e-8)
                torch.testing.assert_close(g.w_time.grad[m], s_.conv_time.weight.grad[:, 0, 0, :], atol=1e-8, rtol=1e-7)
                torch.testing.assert_close(g.w_spat.grad[m], s_.conv_spat.weight.grad[:, :, :, 0], atol=1e-8, rtol=1e-7)
            g.eval()
            for s_ in singles:
                s_.eval()
            torch.testing.assert_close(g(torch.stack(xs, 1))[:, 1], singles[1](xs[1][:, None]), atol=1e-10, rtol=0)
        finally:
            torch.set_default_dtype(torch.float32)


class TrainGeneric(unittest.TestCase):
    def test_crn_and_readouts(self):
        X, y = data(60, dtype=torch.float32)
        base = np.arange(0, 40)
        rows = np.stack([np.r_[base, 50], np.r_[base, 50], np.r_[base, 51]])
        for cls in (dg.EEGNetG, dg.GroupShallow):
            out = dg.train_generic(cls, X, y, rows, np.array([0, 0, 0]), [7], 3, "cpu",
                                   score_idx=np.arange(50, 56), val_idx=np.arange(40, 50), epochs=3, bs=8)
            self.assertTrue(np.array_equal(out["p_final"][0], out["p_final"][1]))  # A/A exact
            self.assertFalse(np.array_equal(out["p_final"][0], out["p_final"][2]))
            self.assertTrue(np.all((out["es_epoch"] >= 1) & (out["es_epoch"] <= 3)))
            self.assertEqual(out["p_es"].shape, (3, 6))

    def test_eegnetg_matches_c2_trainer_final_epoch(self):
        from models import eegnet_group as eg
        X, y = data(60, dtype=torch.float32)
        rows = np.stack([np.r_[np.arange(40), 50], np.r_[np.arange(40), 51]])
        a = dg.train_generic(dg.EEGNetG, X, y, rows, np.array([0, 0]), [9], 4, "cpu",
                             score_idx=np.arange(50, 60), val_idx=np.arange(40, 50), epochs=2, bs=8)
        m = eg.train_group(X, y, rows, np.array([0, 0]), [9], 4, "cpu", epochs=2, bs=8)
        p = eg.predict_proba(m, X, np.arange(50, 60))
        pc = torch.gather(p, 2, y[50:60][None, :, None].expand(2, -1, 1))[..., 0].double().numpy()
        np.testing.assert_allclose(a["p_final"], pc, atol=1e-6)


if __name__ == "__main__":
    unittest.main()


class C6Plan(unittest.TestCase):
    def test_plan_records_analysis(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_r6_c3_synthetic import fake_meta
        from experiments.pmis_r6 import lee2019
        from experiments.pmis_r7 import c6
        from evaluation import r7_c6_analysis as c6_analysis
        meta = fake_meta(n_subj=3)
        tasks = lee2019.build_tasks(meta)
        plan = c6.build_plan(tasks, meta["label"])
        for g in plan["groups"]:
            self.assertEqual(len(g["val"]), 60)
            self.assertEqual(len(g["train"]), 240)
            self.assertFalse(set(g["val"]) & set(g["train"]))
            for pr in g["pairs"]:
                self.assertIn(pr["c"], set(g["train"]))
                self.assertIn(pr["e"], set(g["train"]))
            self.assertEqual(sum(a["arm"] == "AA" for a in g["arms"]), 1)
        rng = np.random.default_rng(0)
        results = {}
        for model in c6.MODELS:
            for g in plan["groups"]:
                pf = rng.uniform(0.2, 0.8, (len(g["arms"]), len(g["T"])))
                pe = rng.uniform(0.2, 0.8, pf.shape)
                for p in (pf, pe):
                    p[[i for i, a in enumerate(g["arms"]) if a["arm"] == "AA"][0]] = p[0]
                results[(model, g["subject"], g["fold"])] = {"p_final": pf, "p_es": pe, "es_epoch": np.full(len(g["arms"]), 40)}
        recs, aa = {}, {}
        for model in c6.MODELS:
            recs[model], aa[model] = c6.pair_records(plan, results, model)
            self.assertEqual(max(aa[model]), 0.0)
        out = c6_analysis.analyse(recs, aa, {r["k"]: 0.05 for r in recs["EEGNet"]})
        self.assertEqual(set(out["family_holm"]), set(c6_analysis.FAMILY))
