"""R9-B synthetic tests: reconstruction metric and decision rule, construction parity with R8, GPU determinism."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
from models import recon  # noqa: E402

CKPT = ROOT / "data" / "r8_e" / "ckpt" / "cbramod_pretrained_weights.pth"


def test_nmse_and_rule():
    import torch

    class Zero(torch.nn.Module):
        def forward(self, x, mask=None):
            return torch.zeros_like(x)

    class Oracle(torch.nn.Module):
        def forward(self, x, mask=None):
            return x.clone()

    X = torch.randn(40, 5, 800)
    m = recon.make_mask(40, 5, 1)
    subj = np.repeat(np.arange(4), 10)
    ez = recon.per_trial_errors(Zero(), X, m, "cpu")
    eo = recon.per_trial_errors(Oracle(), X, m, "cpu")
    assert abs(recon.nmse(ez, subj)["mean_over_participants"] - 1.0) < 1e-12
    assert recon.nmse(eo, subj)["mean_over_participants"] == 0.0
    assert recon.classify(0.5, 1.5, 0.55, 0.5)[0] == "CERTIFIED_ENGAGED"
    assert recon.classify(0.5, 1.5, 0.75, 0.5)[0] == "MONTAGE_DEGRADED"
    assert recon.classify(0.95, 1.5, 0.5, 0.5)[0] == "NOT_ESTABLISHED"
    assert recon.classify(0.5, 0.55, 0.5, 0.5)[0] == "NOT_ESTABLISHED"


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint")
def test_construction_parity_with_r8():
    import torch

    from models import models_de
    from models import models_r9
    torch.manual_seed(5)
    a = models_de.make_cbramod_ft(CKPT)
    torch.manual_seed(5)
    b = models_r9.make_cbramod(CKPT, 20)
    torch.manual_seed(5)
    c = models_r9.make_cbramod(None, 20)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)  # R9 pretrained == R8 model exactly
    head = [k for k in sa if k.startswith("classifier")]
    assert head and all(torch.equal(sa[k], sc[k]) for k in head)  # same head initialisation
    bbk = [k for k in sa if k.startswith("backbone") and "weight" in k]
    assert any(not torch.equal(sa[k], sc[k]) for k in bbk)  # backbone differs (random vs pretrained)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    pre = models_r9.make_backbone(CKPT, 0, keep_head=True).state_dict()
    assert set(pre) == set(ck) and all(torch.equal(pre[k], ck[k]) for k in ck)


@pytest.mark.skipif(not __import__("torch").cuda.is_available() or not CKPT.exists(), reason="GPU + checkpoint")
def test_gpu_determinism_random_init_and_recon():
    import torch

    from models import models_de, seq_train
    from models import models_r9
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    X = (torch.randn(120, 20, 800, generator=g) * 0.2).to(dev)
    y = torch.as_tensor(np.tile([0, 1], 60), device=dev)
    rows = np.arange(80)
    kw = dict(score_idx=np.arange(100, 110), val_idx=np.arange(80, 100), epochs=2, bs=32, device=dev, label_smoothing=0.1, clip=1.0)
    r1 = seq_train.train_arm(lambda: models_r9.make_cbramod(None, 20), models_de.cbramod_opt, X, y, rows, 11, 12, **kw)
    r2 = seq_train.train_arm(lambda: models_r9.make_cbramod(None, 20), models_de.cbramod_opt, X, y, rows, 11, 12, **kw)
    assert np.array_equal(r1["p_final"], r2["p_final"]) and np.array_equal(r1["p_es"], r2["p_es"])
    bb = models_r9.make_backbone(CKPT, 0, keep_head=True).to(dev)
    m = recon.make_mask(120, 20, 9302)
    e1 = recon.per_trial_errors(bb, X, m, dev)
    e2 = recon.per_trial_errors(bb, X, m, dev)
    assert all(np.array_equal(e1[k], e2[k]) for k in e1)
