"""R10-A grouped deep training without a validation set (final-epoch read-out, the R4 / R8-B / R9-C recipe).

train_group_final reproduces the training mechanics of pmis_r7.deep_group.train_generic exactly (per-pair initialisation,
per-pair row-position permutation, per-pair shared dropout masks drawn from one group stream, Adam, per-model batch-mean
loss, max-norm hook) but performs no per-epoch evaluation, because R4's arms leave no validation set. Its final-epoch
probabilities equal train_generic's p_final bit for bit (tests/test_r10_a.py); evaluation never consumes RNG.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.deep_group import _proba, draw_masks


def train_group_final(cls, X, y, rows, pair_of, pair_seeds, group_seed, device, epochs: int = 100, bs: int = 64,
                      lr: float = 1e-3):
    M, n_rows = rows.shape
    P = len(pair_seeds)
    C, T = X.shape[1], X.shape[2]
    model = cls(M, C, T).to(device)
    for p_idx, s in enumerate(pair_seeds):
        torch.manual_seed(int(s))
        single = cls.make_single(C, T)
        for m in np.flatnonzero(pair_of == p_idx):
            model.load_single(int(m), single)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rows_t = torch.as_tensor(rows, device=device, dtype=torch.long)
    pair_t = torch.as_tensor(pair_of, device=device, dtype=torch.long)
    gens = [torch.Generator(device="cpu").manual_seed(int(s) + 7_919) for s in pair_seeds]
    mgen = torch.Generator(device=device).manual_seed(int(group_seed))
    model.train()
    for _ in range(epochs):
        perms = torch.stack([torch.randperm(n_rows, generator=g) for g in gens]).to(device)
        order = torch.gather(rows_t, 1, perms[pair_t])
        for i in range(0, n_rows, bs):
            idx = order[:, i:i + bs]
            B = idx.shape[1]
            if B < 2:
                continue
            logits = model(X[idx].permute(1, 0, 2, 3), draw_masks(model, B, P, mgen, device), pair_t)
            loss = F.cross_entropy(logits.permute(1, 2, 0), y[idx], reduction="none").mean(dim=1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.max_norm()
    return model


def correct_class(model, X, y, idx):
    """(M, n) correct-class probabilities and (M, n) 0/1 decisions on idx (eval mode, running BN statistics)."""
    pr = _proba(model, X, idx)  # (M, n, 2)
    yt = y[torch.as_tensor(idx, device=X.device, dtype=torch.long)]
    p = torch.gather(pr, 2, yt[None, :, None].expand(pr.shape[0], -1, 1))[..., 0]
    c = (pr.argmax(-1) == yt[None]).double()
    return p.double().cpu().numpy(), c.cpu().numpy(), pr
