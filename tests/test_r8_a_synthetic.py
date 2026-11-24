"""R8-A (Shin2022 natural-duplicate PMIS) code on SYNTHETIC windows and a SYNTHETIC registry only."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from models.vendor_r4.r4_models import make_eegnet  # noqa: E402
from experiments.pmis_r8 import shin
from evaluation import r8_shin_analysis as shin_analysis
from models import shin_models  # noqa: E402


def fake(n_sub=3, n_rec=4, n_trials=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, reg = [], []
    for s in range(1, n_sub + 1):
        for r in range(n_rec):
            sha = f"{s:02d}{r:02d}" + "0" * 60
            reg.append({"annex_sha256": sha, "subject": f"S{s:02d}"})
            if r == 0:
                reg.append({"annex_sha256": sha, "subject": f"S{s:02d}"})  # alias path of recording 0
            for k in range(n_trials):
                rows.append((sha, k, f"S{s:02d}", 1 + (k % 2), True))
    sha, blk, sub, lab, main = (np.array(c) for c in zip(*rows))
    n = len(rows)
    a = rng.standard_normal((n, 16, 48))
    cov = a @ np.swapaxes(a, 1, 2) / 48 + 0.1 * np.eye(16)
    cov[lab == 2] += 0.3 * np.eye(16)
    eeg = rng.standard_normal((n, 16, 240)).astype(np.float32)
    return {"sha": sha, "block": blk, "subject": sub, "label": lab, "b_main": main, "cov830_FB_EARLY": cov,
            "eeg440_FB_EARLY": eeg}, reg


class Tasks(unittest.TestCase):
    def test_build(self):
        z, reg = fake()
        tasks = shin.build_tasks(z, reg)
        self.assertEqual(len(tasks), 3)
        for t in tasks:
            self.assertEqual(t["T"].size, 12)
            self.assertEqual(len(np.intersect1d(t["T"], t["base"])), 0)
            self.assertEqual(len(t["pairs"]), 2 * 15)  # C(6,2) per class
            for p in t["pairs"]:
                self.assertEqual(z["label"][p["a"]], z["label"][p["b"]])
                self.assertIn(p["c"], set(t["base"].tolist()))
                self.assertNotEqual(p["c"], p["e"])
            rows, extras = shin.arm_rows(t)
            self.assertEqual(rows.shape, (24, t["base"].size + 1))
        self.assertEqual(shin.commitment(tasks), shin.commitment(shin.build_tasks(z, reg)))

    def test_tslr_and_analysis(self):
        z, reg = fake()
        tasks = shin.build_tasks(z, reg)
        y = z["label"].astype(np.int64)
        recs = []
        for t in tasks:
            P, NT, _ = shin_models.tslr_task(t, z["cov830_FB_EARLY"], y)
            self.assertEqual(NT.shape, (t["T"].size, t["T"].size))
            _, extras = shin.arm_rows(t)
            pos = {int(v): i for i, v in enumerate(t["T"].tolist())}
            for pr, a in zip(t["pairs"], shin.pair_records(t, P, extras, y)):
                ia, ib = pos[pr["a"]], pos[pr["b"]]
                r = {"subject": a["subject"], "label": a["label"], "pos_dist": a["pos_dist"], "sign": a["sign"]}
                r.update({"t_" + k: v for k, v in a.items() if k.startswith("p")})
                r.update({"e_" + k: v for k, v in a.items() if k.startswith("p")})
                r.update({"nt_pAA": NT[ia, ia], "nt_pBA": NT[ia, ib], "nt_pAB": NT[ib, ia], "nt_pBB": NT[ib, ib]})
                recs.append(r)
        out = shin_analysis.analyse(recs, 0.0)
        self.assertEqual(set(out["family_holm"]), set(shin_analysis.FAMILY))
        self.assertGreater(out["family_holm"]["H1_TSLR_d"]["mean_points"], 0)


class EEGNet250(unittest.TestCase):
    def test_grouped_equals_single_and_aa(self):
        torch.set_default_dtype(torch.float64)
        try:
            torch.manual_seed(3)
            s = make_eegnet(16, 240, kern=125)
            g = shin_models.EEGNet250(2, 16, 240).double()
            g.load_single(0, s)
            g.load_single(1, s)
            x = torch.randn(5, 16, 240)
            g.eval()
            s.eval()
            out = g(torch.stack([x, x], 1))
            torch.testing.assert_close(out[:, 0], s(x[:, None]), atol=1e-10, rtol=0)
        finally:
            torch.set_default_dtype(torch.float32)
        z, reg = fake(n_sub=1)
        t = shin.build_tasks(z, reg)[0]
        X = torch.as_tensor(z["eeg440_FB_EARLY"])
        y01 = torch.as_tensor(z["label"].astype(np.int64) - 1)
        import models.deep_group as dg
        rows, _ = shin.arm_rows(t)
        rows = np.vstack([rows, rows[:1]])
        o = dg.train_generic(shin_models.EEGNet250, X, y01, rows, np.zeros(rows.shape[0], dtype=np.int64), [1], 2, "cpu",
                             score_idx=t["T"], val_idx=t["T"][:2], epochs=2, bs=16)
        # CPU batched BLAS is not bit-identical across batch slots; the GPU run enforces exact A/A (fail closed)
        np.testing.assert_allclose(o["p_final"][-1], o["p_final"][0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
