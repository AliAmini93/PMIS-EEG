"""M independent EEGNet-8,2 models trained simultaneously as one grouped network (R6-C2).

Each model is exactly the R4/Paper A EEGNet-8,2 (`vendor_r4.r4_models.make_eegnet`, kern 64 at 128 Hz, F1 8, D 2,
F2 16, dropout 0.25, max-norm 1.0 / 0.25, Adam 1e-3, batch 64, 100 epochs, no early stopping). Grouping is an
implementation device only: every layer is block-diagonal over models (grouped convolutions, per-model
BatchNorm channels, per-model linear head), the loss is the sum of per-model batch-mean cross-entropies and Adam
is element-wise, so each model's trajectory equals that of a separately trained model given the same
initialisation, row order and dropout masks (verified in tests/test_r6_c2_synthetic.py).

Common random numbers (PMIS arms): all arms of one *pair* share the initialisation (torch.manual_seed(pair seed)
on the single-model constructor), the per-epoch row permutation (a per-pair generator over row positions; the
swapped row occupies the same position in every arm) and every dropout mask. Arms of a pair therefore differ
only in the content of one training row.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vendor_r4.r4_models import make_eegnet

F1, D, F2, KERN, P_DROP = 8, 2, 16, 64, 0.25


class GroupEEGNet(nn.Module):
    def __init__(self, M: int, C: int, T: int, kern: int = KERN):
        super().__init__()
        self.M, self.C, self.T, self.kern = M, C, T, kern
        self.conv1 = nn.Conv2d(M, M * F1, (1, kern), padding=(0, kern // 2), groups=M, bias=False)
        self.bn1 = nn.BatchNorm2d(M * F1)
        self.dw = nn.Conv2d(M * F1, M * F1 * D, (C, 1), groups=M * F1, bias=False)
        self.bn2 = nn.BatchNorm2d(M * F1 * D)
        self.sep_dw = nn.Conv2d(M * F1 * D, M * F1 * D, (1, 16), padding=(0, 8), groups=M * F1 * D, bias=False)
        self.sep_pw = nn.Conv2d(M * F1 * D, M * F2, 1, groups=M, bias=False)
        self.bn3 = nn.BatchNorm2d(M * F2)
        with torch.no_grad():
            t1 = T + 2 * (kern // 2) - kern + 1
            t2 = t1 // 4
            t3 = t2 + 16 - 16 + 1
            self.t_final = t3 // 8
        self.n_feat = F2 * self.t_final
        self.fc_w = nn.Parameter(torch.zeros(M, 2, self.n_feat))
        self.fc_b = nn.Parameter(torch.zeros(M, 2))
        self.t_drop1, self.t_drop2 = t2, self.t_final

    # ---------------------------------------------------------------- parameter transfer (single <-> group)
    def load_single(self, m: int, single: nn.Module) -> None:
        sd = single.state_dict()
        with torch.no_grad():
            self.conv1.weight[m * F1:(m + 1) * F1] = sd["b1.0.weight"]
            for g, key in ((self.bn1, "b1.1"), (self.bn2, "b2.0"), (self.bn3, "b3.2")):
                n = sd[f"{key}.weight"].numel()
                g.weight[m * n:(m + 1) * n] = sd[f"{key}.weight"]
                g.bias[m * n:(m + 1) * n] = sd[f"{key}.bias"]
                g.running_mean[m * n:(m + 1) * n] = sd[f"{key}.running_mean"]
                g.running_var[m * n:(m + 1) * n] = sd[f"{key}.running_var"]
            self.dw.weight[m * F1 * D:(m + 1) * F1 * D] = sd["dw.weight"]
            self.sep_dw.weight[m * F1 * D:(m + 1) * F1 * D] = sd["b3.0.weight"]
            self.sep_pw.weight[m * F2:(m + 1) * F2] = sd["b3.1.weight"]
            self.fc_w[m] = sd["fc.weight"]
            self.fc_b[m] = sd["fc.bias"]

    def max_norm(self) -> None:
        with torch.no_grad():
            self.dw.weight.data = torch.renorm(self.dw.weight.data, 2, 0, 1.0)
            w = self.fc_w.data.reshape(self.M * 2, self.n_feat)
            self.fc_w.data = torch.renorm(w, 2, 0, 0.25).reshape(self.M, 2, self.n_feat)

    # ---------------------------------------------------------------- forward
    impl = "fast"  # "fast": FFT / einsum / shifted sums (same parameters); "conv": grouped nn.Conv2d (reference)

    def _front(self, x):
        """conv1 -> BN1 -> depthwise spatial conv; returns (B, M*F1*D, 1, t1)."""
        B, M = x.shape[0], self.M
        if self.impl == "conv":
            return self.dw(self.bn1(self.conv1(x)))
        k = self.kern
        xp = F.pad(x, (k // 2, k // 2))  # (B, M, C, T + k)
        n = xp.shape[-1] + k  # no circular wrap for the kept lags
        t1 = xp.shape[-1] - k + 1
        xf = torch.fft.rfft(xp, n=n)  # (B, M, C, nf)
        wf = torch.fft.rfft(self.conv1.weight.view(M, F1, k), n=n)  # (M, F1, nf)
        z = torch.fft.irfft(xf[:, :, None] * torch.conj(wf)[None, :, :, None], n=n)[..., :t1]  # (B, M, F1, C, t1)
        z = self.bn1(z.reshape(B, M * F1, self.C, t1)).view(B, M, F1, self.C, t1)
        w = self.dw.weight.view(M, F1, D, self.C)
        return torch.einsum("bmfct,mfdc->bmfdt", z, w).reshape(B, M * F1 * D, 1, t1)

    def _separable(self, h):
        """depthwise temporal (1, 16) conv, padding 8, then per-model pointwise conv; returns (B, M*F2, 1, t)."""
        if self.impl == "conv":
            return self.sep_pw(self.sep_dw(h))
        B, M = h.shape[0], self.M
        hp = F.pad(h[:, :, 0], (8, 8))  # (B, M*F1*D, t + 16)
        t3 = hp.shape[-1] - 16 + 1
        w = self.sep_dw.weight.view(M * F1 * D, 16)
        out = hp[..., 0:t3] * w[None, :, 0:1]
        for j in range(1, 16):
            out = out + hp[..., j:j + t3] * w[None, :, j:j + 1]
        pw = self.sep_pw.weight.view(M, F2, F1 * D)
        return torch.einsum("bmit,moi->bmot", out.view(B, M, F1 * D, t3), pw).reshape(B, M * F2, 1, t3)

    def forward(self, x: torch.Tensor, masks=None, pair_of=None) -> torch.Tensor:
        """x: (B, M, C, T). masks: None (eval / no dropout) or (m1, m2) with shapes (B, P, F1*D, 1, t_drop1) and
        (B, P, F2, 1, t_drop2), already scaled by 1/(1-p); pair_of: (M,) long, the pair of each model."""
        B, M = x.shape[0], self.M
        h = self._front(x)
        h = F.avg_pool2d(F.elu(self.bn2(h)), (1, 4))
        if masks is not None:
            h = (h.view(B, M, F1 * D, 1, -1) * masks[0][:, pair_of]).view(B, M * F1 * D, 1, -1)
        h = self._separable(h)
        h = F.avg_pool2d(F.elu(self.bn3(h)), (1, 8))
        if masks is not None:
            h = (h.view(B, M, F2, 1, -1) * masks[1][:, pair_of]).view(B, M * F2, 1, -1)
        f = h.reshape(B, M, self.n_feat)
        return torch.einsum("bmf,mkf->bmk", f, self.fc_w) + self.fc_b[None]


def draw_masks(B: int, P: int, gen: torch.Generator, device, t1: int, t2: int, p: float = P_DROP):
    keep = 1.0 - p
    m1 = (torch.rand((B, P, F1 * D, 1, t1), generator=gen, device=device) < keep).float() / keep
    m2 = (torch.rand((B, P, F2, 1, t2), generator=gen, device=device) < keep).float() / keep
    return m1, m2


def train_group(X: torch.Tensor, y: torch.Tensor, rows: np.ndarray, pair_of: np.ndarray, pair_seeds: list[int],
                group_seed: int, device, epochs: int = 100, bs: int = 64, lr: float = 1e-3,
                dropout: bool = True, max_batches: int | None = None) -> GroupEEGNet:
    """rows: (M, n_rows) dataset indices per arm (equal n_rows). pair_of: (M,) pair index in [0, P).
    pair_seeds[P]: CRN seed of each pair (initialisation and row order). group_seed: dropout-mask stream."""
    M, n_rows = rows.shape
    P = len(pair_seeds)
    C, T = X.shape[1], X.shape[2]
    model = GroupEEGNet(M, C, T).to(device)
    for p_idx, s in enumerate(pair_seeds):
        torch.manual_seed(int(s))
        single = make_eegnet(C, T, kern=KERN)
        for m in np.flatnonzero(pair_of == p_idx):
            model.load_single(int(m), single)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rows_t = torch.as_tensor(rows, device=device, dtype=torch.long)
    pair_t = torch.as_tensor(pair_of, device=device, dtype=torch.long)
    gens = [torch.Generator(device="cpu").manual_seed(int(s) + 7_919) for s in pair_seeds]
    mgen = torch.Generator(device=device).manual_seed(int(group_seed))
    model.train()
    done = 0
    for _ in range(epochs):
        perms = torch.stack([torch.randperm(n_rows, generator=g) for g in gens]).to(device)  # (P, n_rows)
        order = torch.gather(rows_t, 1, perms[pair_t])  # (M, n_rows): same positions within a pair
        for i in range(0, n_rows, bs):
            idx = order[:, i:i + bs]
            B = idx.shape[1]
            if B < 2:
                continue
            xb = X[idx].permute(1, 0, 2, 3)  # (B, M, C, T)
            yb = y[idx]  # (M, B)
            masks = draw_masks(B, P, mgen, device, model.t_drop1, model.t_drop2) if dropout else None
            logits = model(xb, masks, pair_t)  # (B, M, 2)
            loss = F.cross_entropy(logits.permute(1, 2, 0), yb, reduction="none").mean(dim=1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.max_norm()
            done += 1
            if max_batches is not None and done >= max_batches:
                return model
    return model


@torch.no_grad()
def predict_proba(model: GroupEEGNet, X: torch.Tensor, idx: np.ndarray, chunk: int = 32) -> torch.Tensor:
    """Softmax probabilities of every model on the same trials idx: returns (M, len(idx), 2)."""
    model.eval()
    out = []
    it = torch.as_tensor(idx, device=X.device, dtype=torch.long)
    for i in range(0, it.numel(), chunk):
        xb = X[it[i:i + chunk]][:, None].expand(-1, model.M, -1, -1)  # (B, M, C, T)
        out.append(torch.softmax(model(xb), dim=-1).permute(1, 0, 2))
    return torch.cat(out, dim=1)


@torch.no_grad()
def predict_own(model: GroupEEGNet, X: torch.Tensor, rows: np.ndarray, chunk: int = 32) -> torch.Tensor:
    """Each model on its own rows (training-set fit): rows (M, n) -> probabilities (M, n, 2)."""
    model.eval()
    r = torch.as_tensor(rows, device=X.device, dtype=torch.long)
    out = []
    for i in range(0, r.shape[1], chunk):
        xb = X[r[:, i:i + chunk]].permute(1, 0, 2, 3)
        out.append(torch.softmax(model(xb), dim=-1).permute(1, 0, 2))
    return torch.cat(out, dim=1)
