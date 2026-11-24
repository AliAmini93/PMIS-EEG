"""R9-B1: masked-patch reconstruction fidelity of CBraMod on Lee2019 inputs (forward passes only; frozen before running).

Pretraining objective (pretrain_trainer.py @ b9e9610): x / 100, Bernoulli(0.5) patch mask, masked patches replaced by the
(zero) mask encoding, MSE between reconstruction and input on masked patches. Fidelity metric: normalised masked error
NMSE = sum over masked patches (y - x)^2 / sum over masked patches x^2 (the trivial zero predictor scores 1). Per
participant (ratio of sums), cohort = mean over participants (equal weights)."""

from __future__ import annotations

import numpy as np


def make_mask(n, c, seed):
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.rand((n, c, 4), generator=g) < 0.5).long()


def per_trial_errors(model, X, mask, device, chunk=64):
    """X: (N, C, 800) float tensor already divided by 100 (on device); mask (N, C, 4) long (cpu).
    Returns numpy arrays (N,) of masked SE, masked SS, unmasked SE, unmasked SS."""
    import torch
    out = {k: [] for k in ("se_m", "ss_m", "se_u", "ss_u")}
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            x = X[i:i + chunk].view(-1, X.shape[1], 4, 200)
            m = mask[i:i + chunk].to(device)
            y = model(x, mask=m)
            e2, s2 = ((y - x) ** 2).sum(-1), (x ** 2).sum(-1)  # (B, C, 4)
            mm = m.to(e2.dtype)
            out["se_m"].append((e2 * mm).sum((1, 2)).double().cpu())
            out["ss_m"].append((s2 * mm).sum((1, 2)).double().cpu())
            out["se_u"].append((e2 * (1 - mm)).sum((1, 2)).double().cpu())
            out["ss_u"].append((s2 * (1 - mm)).sum((1, 2)).double().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def nmse(err, subject, keep=None, which="m"):
    keep = np.ones(subject.size, dtype=bool) if keep is None else keep
    vals = []
    for p in np.unique(subject):
        m = (subject == p) & keep
        if m.sum():
            vals.append(err[f"se_{which}"][m].sum() / err[f"ss_{which}"][m].sum())
    v = np.array(vals)
    return {"mean_over_participants": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "n_participants": int(v.size), "n_trials": int(keep.sum())}


def classify(n_pre19_clean, n_rand19_clean_min, n_pre20, n_pre19_all):
    """Declared decision rule (frozen). V1: the checkpoint reconstructs pretraining-format Lee2019 input better than the zero
    predictor by >= 10 % and better than every random initialisation by >= 0.10. V2: on the R8 input the pretrained model
    is also >= 10 % better than zero and within 0.10 NMSE of its pretraining-format performance."""
    v1 = (n_pre19_clean <= 0.90) and (n_pre19_clean <= n_rand19_clean_min - 0.10)
    v2 = (n_pre20 <= 0.90) and (n_pre20 <= n_pre19_all + 0.10)
    if not v1:
        return "NOT_ESTABLISHED", v1, v2
    return ("CERTIFIED_ENGAGED" if v2 else "MONTAGE_DEGRADED"), v1, v2
