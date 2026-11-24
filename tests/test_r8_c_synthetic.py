"""R8-C ceiling audit on SYNTHETIC records: a pure ceiling world and a substantive world are classified as declared."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from evaluation import r8_ceiling as ceiling  # noqa: E402


def world(substantive, n_part=40, n_pairs=60, seed=0):
    rng = np.random.default_rng(seed)
    rec = {k: [] for k in ("subject", "label", "pAA", "pAB", "pBA", "pBB")}
    for s in range(n_part):
        skill = rng.uniform(-1, 3)  # participant decodability on the logit scale
        gain = 1.0 + (1.5 - 0.5 * skill if substantive else 0.0)  # logit gain from the own copy
        for k in range(n_pairs):
            la, lb = skill + rng.normal(0, 1, 2)
            sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
            rec["subject"].append(s)
            rec["label"].append(1 + k % 2)
            rec["pAB"].append(sig(la)); rec["pAA"].append(sig(la + gain))
            rec["pBA"].append(sig(lb)); rec["pBB"].append(sig(lb + gain))
    return {k: np.array(v) for k, v in rec.items()}


class Ceiling(unittest.TestCase):
    def test_pure_ceiling_world(self):
        out = ceiling.audit(world(False))
        self.assertLess(out["r_dprob_BA"]["r"], -0.6)  # probability scale shows the spurious pattern
        self.assertGreater(out["logit_eps0.001"]["r_dlogit_BA"]["r"], -0.3)  # constant logit gain
        self.assertEqual(out["verdict"], "LARGELY_CEILING_ARTEFACT")

    def test_substantive_world(self):
        out = ceiling.audit(world(True))
        self.assertLess(out["logit_eps0.001"]["r_dlogit_BA"]["r"], -0.5)
        self.assertEqual(out["verdict"], "SUBSTANTIVE")


if __name__ == "__main__":
    unittest.main()
