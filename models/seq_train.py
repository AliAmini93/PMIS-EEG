"""Sequential exact-refit trainer with common random numbers (CRN) for models that cannot be grouped (R8-D, R8-E).

Every arm m of a group is trained from scratch, one after another, with
  initialisation  torch.manual_seed(init_seed[pair])  then  make_model()      (identical within a pair)
  row order       torch.Generator('cpu').manual_seed(init_seed[pair] + 7919): per-epoch permutation of row POSITIONS,
                  so a pair's swapped row occupies the same batch slot in both arms
  dropout stream  torch.manual_seed(group_seed) after construction: all arms of a group have identical shapes and
                  batch sizes, hence identical RNG consumption and identical dropout / drop-path masks per step
Read-outs (same semantics as pmis_r7.deep_group.train_generic): correct-class probability on score_idx after the final
epoch, and at the epoch of minimum validation cross-entropy (val_idx never in training; ties keep the earliest epoch).
With torch.use_deterministic_algorithms(True) on CUDA a repeated arm is bit-identical (A/A check).
"""

from __future__ import annotations

import numpy as np


def train_arm(make_model, make_opt, X, y, rows, init_seed, group_seed, score_idx, val_idx, epochs, bs, device,
              label_smoothing=0.0, clip=None, post_step=None, eval_chunk=128):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(int(init_seed))
    model = make_model().to(device)
    n = int(rows.size)
    steps_per_epoch = sum(1 for i in range(0, n, bs) if min(bs, n - i) >= 2)
    opt, sched, per = make_opt(model, steps_per_epoch, epochs)
    gen = torch.Generator(device="cpu").manual_seed(int(init_seed) + 7_919)
    torch.manual_seed(int(group_seed))
    rows_t = torch.as_tensor(rows, device=device, dtype=torch.long)
    s_idx = torch.as_tensor(score_idx, device=device, dtype=torch.long)
    v_idx = torch.as_tensor(val_idx, device=device, dtype=torch.long)
    ys, yv = y[s_idx], y[v_idx]

    @torch.no_grad()
    def proba(idx):
        model.eval()
        out = torch.cat([torch.softmax(model(X[idx[i:i + eval_chunk]]).float(), -1) for i in range(0, idx.numel(), eval_chunk)])
        model.train()
        return out

    best_loss, best_ep, best_p = float("inf"), 0, None
    model.train()
    for ep in range(epochs):
        order = rows_t[torch.randperm(n, generator=gen).to(device)]
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            if idx.numel() < 2:
                continue
            loss = F.cross_entropy(model(X[idx]), y[idx], label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if post_step is not None:
                post_step(model)
            if sched is not None and per == "step":
                sched.step()
        if sched is not None and per == "epoch":
            sched.step()
        pv = proba(v_idx)
        vloss = float(-torch.log(pv.gather(1, yv[:, None])[:, 0].clamp_min(1e-12)).mean())
        pc = proba(s_idx).gather(1, ys[:, None])[:, 0]
        if vloss < best_loss:
            best_loss, best_ep, best_p = vloss, ep + 1, pc.clone()
    return {"p_final": pc.double().cpu().numpy(), "p_es": best_p.double().cpu().numpy(), "es_epoch": best_ep,
            "val_loss_min": best_loss}


def run_group(make_model, make_opt, g, X, y01, device, epochs, bs, **kw):
    """C6-format group g (pmis_r7.c6.build_plan): rows = train + [extra] for every arm; returns stacked read-outs."""
    train = np.asarray(g["train"], dtype=np.int64)
    seeds = [p["seed"] for p in g["pairs"]]
    res = [train_arm(make_model, make_opt, X, y01, np.concatenate([train, [a["extra"]]]).astype(np.int64),
                     seeds[a["pair"]], g["group_seed"], np.asarray(g["T"], dtype=np.int64),
                     np.asarray(g["val"], dtype=np.int64), epochs, bs, device, **kw) for a in g["arms"]]
    out = {"p_final": np.stack([r["p_final"] for r in res]), "p_es": np.stack([r["p_es"] for r in res]),
           "es_epoch": np.array([r["es_epoch"] for r in res]), "val_loss_min": np.array([r["val_loss_min"] for r in res])}
    for k in ("p_final", "p_es"):
        if not np.all(np.isfinite(out[k])):
            raise FloatingPointError("non-finite output")
    return out
