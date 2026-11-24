"""R7-C6: grouped deep decoders with common random numbers, early-stopping read-out, and a ShallowConvNet.

* EEGNetG: the unmodified C2 GroupEEGNet (exact R4 EEGNet-8,2), with a mask-shape interface.
* ShallowSingle: reference ShallowConvNet (Schirrmeister et al. 2017) adapted to 128 Hz:
  temporal conv (40 filters, kernel 13 ~ 25 at 250 Hz, bias) -> spatial conv (40 x C, no bias) -> BatchNorm ->
  square -> average pool (1, 37) stride (1, 7) -> log(clamp 1e-6) -> dropout 0.5 -> dense (2 classes).
* GroupShallow: M ShallowSingles as one network. conv_time and conv_spat are linear with nothing in between, so their
  composite V[k, c, tau] = sum_f W_spat[k, f, c] W_time[f, tau] is applied as one grouped Conv1d (C -> 40 per model);
  the conv_time bias becomes the constant sum_{f,c} W_spat[k, f, c] b_time[f]. Exactly equal to the sequential model
  (tests/test_r7_deep_group.py).
* train_generic: the C2 training mechanics (per-pair init, per-pair row order, per-pair shared dropout masks, Adam,
  per-model batch-mean loss), plus, after every epoch, the probabilities on the scored trials and the validation loss;
  read-outs: final epoch and the epoch of minimum validation loss (early stopping by selection).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vendor_r4.r4_models import make_eegnet
from models.eegnet_group import D, F1, F2, KERN, GroupEEGNet

K_T, N_F, POOL, STRIDE, P_SH = 13, 40, 37, 7, 0.5


class EEGNetG(GroupEEGNet):
    p_drop = 0.25

    def mask_shapes(self):
        return [(F1 * D, self.t_drop1), (F2, self.t_drop2)]

    @staticmethod
    def make_single(C, T):
        return make_eegnet(C, T, kern=KERN)


class ShallowSingle(nn.Module):
    def __init__(self, C: int, T: int):
        super().__init__()
        self.conv_time = nn.Conv2d(1, N_F, (1, K_T))
        self.conv_spat = nn.Conv2d(N_F, N_F, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(N_F)
        self.drop = nn.Dropout(P_SH)
        self.t_out = (T - K_T + 1 - POOL) // STRIDE + 1
        self.fc = nn.Linear(N_F * self.t_out, 2)

    def forward(self, x):  # x: (B, 1, C, T)
        h = self.bn(self.conv_spat(self.conv_time(x)))
        h = F.avg_pool2d(h * h, (1, POOL), (1, STRIDE))
        h = self.drop(torch.log(torch.clamp(h, min=1e-6)))
        return self.fc(h.flatten(1))


class GroupShallow(nn.Module):
    p_drop = P_SH

    def __init__(self, M: int, C: int, T: int):
        super().__init__()
        self.M, self.C, self.T = M, C, T
        self.w_time = nn.Parameter(torch.zeros(M, N_F, K_T))
        self.b_time = nn.Parameter(torch.zeros(M, N_F))
        self.w_spat = nn.Parameter(torch.zeros(M, N_F, N_F, C))  # [k, f, c]
        self.bn = nn.BatchNorm2d(M * N_F)
        self.t_out = (T - K_T + 1 - POOL) // STRIDE + 1
        self.fc_w = nn.Parameter(torch.zeros(M, 2, N_F * self.t_out))
        self.fc_b = nn.Parameter(torch.zeros(M, 2))

    @staticmethod
    def make_single(C, T):
        return ShallowSingle(C, T)

    def mask_shapes(self):
        return [(N_F, self.t_out)]

    def load_single(self, m: int, s: ShallowSingle) -> None:
        with torch.no_grad():
            self.w_time[m] = s.conv_time.weight[:, 0, 0, :]
            self.b_time[m] = s.conv_time.bias
            self.w_spat[m] = s.conv_spat.weight[:, :, :, 0]
            sl = slice(m * N_F, (m + 1) * N_F)
            self.bn.weight[sl] = s.bn.weight
            self.bn.bias[sl] = s.bn.bias
            self.bn.running_mean[sl] = s.bn.running_mean
            self.bn.running_var[sl] = s.bn.running_var
            self.fc_w[m] = s.fc.weight
            self.fc_b[m] = s.fc.bias

    def max_norm(self) -> None:  # ShallowConvNet has no max-norm constraint
        return None

    def forward(self, x, masks=None, pair_of=None):  # x: (B, M, C, T)
        B, M = x.shape[0], self.M
        V = torch.einsum("mkfc,mft->mkct", self.w_spat, self.w_time).reshape(M * N_F, self.C, K_T)
        b = torch.einsum("mkfc,mf->mk", self.w_spat, self.b_time).reshape(M * N_F)
        h = F.conv1d(x.reshape(B, M * self.C, self.T), V, b, groups=M)  # (B, M*N_F, T-K_T+1)
        h = self.bn(h[:, :, None, :])
        h = F.avg_pool2d(h * h, (1, POOL), (1, STRIDE))
        h = torch.log(torch.clamp(h, min=1e-6))
        if masks is not None:
            h = (h.view(B, M, N_F, 1, -1) * masks[0][:, pair_of]).view(B, M * N_F, 1, -1)
        f = h.reshape(B, M, N_F * self.t_out)
        return torch.einsum("bmf,mkf->bmk", f, self.fc_w) + self.fc_b[None]


def draw_masks(model, B, P, gen, device):
    keep = 1.0 - model.p_drop
    return tuple((torch.rand((B, P, ch, 1, t), generator=gen, device=device) < keep).float() / keep
                 for ch, t in model.mask_shapes())


@torch.no_grad()
def _proba(model, X, idx, chunk=32):
    model.eval()
    it = torch.as_tensor(idx, device=X.device, dtype=torch.long)
    out = []
    for i in range(0, it.numel(), chunk):
        xb = X[it[i:i + chunk]][:, None].expand(-1, model.M, -1, -1)
        out.append(torch.softmax(model(xb), dim=-1).permute(1, 0, 2))
    model.train()
    return torch.cat(out, dim=1)  # (M, n, 2)


def train_generic(cls, X, y, rows, pair_of, pair_seeds, group_seed, device, score_idx, val_idx,
                  epochs: int = 100, bs: int = 64, lr: float = 1e-3) -> dict:
    """Returns correct-class probabilities on score_idx at the final epoch and at each model's minimum-validation-loss
    epoch, plus that epoch. rows: (M, n_rows); val_idx: validation trials (never in rows).
    With val_idx=None only the final-epoch read-out is computed."""
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
    ys = y[torch.as_tensor(score_idx, device=device)]
    yv = None if val_idx is None else y[torch.as_tensor(val_idx, device=device)]
    best_loss = torch.full((M,), float("inf"), device=device)
    best_epoch = torch.zeros(M, dtype=torch.long, device=device)
    best_p = torch.zeros(M, len(score_idx), device=device)
    model.train()
    for ep in range(epochs):
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
        if yv is not None:
            pv = _proba(model, X, val_idx)
            vloss = -torch.log(torch.gather(pv, 2, yv[None, :, None].expand(M, -1, 1))[..., 0].clamp_min(1e-12)).mean(1)
            ps = _proba(model, X, score_idx)
            pc = torch.gather(ps, 2, ys[None, :, None].expand(M, -1, 1))[..., 0]
            better = vloss < best_loss
            best_loss = torch.where(better, vloss, best_loss)
            best_epoch = torch.where(better, torch.full_like(best_epoch, ep + 1), best_epoch)
            best_p = torch.where(better[:, None], pc, best_p)
    if yv is None:
        ps = _proba(model, X, score_idx)
        pc = torch.gather(ps, 2, ys[None, :, None].expand(M, -1, 1))[..., 0]
        return {"p_final": pc.double().cpu().numpy()}
    return {"p_final": pc.double().cpu().numpy(), "p_es": best_p.double().cpu().numpy(),
            "es_epoch": best_epoch.cpu().numpy(), "val_loss_min": best_loss.double().cpu().numpy()}
