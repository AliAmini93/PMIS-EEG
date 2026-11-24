"""Frozen models, copied unchanged from the earlier model implementation: Paper A TS+LR (numpy tangent space;
pyriemann absent from the fixed venv) and EEGNet-8,2. R4 appends batched, multiplicity-weighted equivalents at the
end (verified against the loop versions in tests/test_r4_g3.py); the R3 functions are not modified."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------------ preprocessing (Paper A settings)
def causal_bandpass(x_ch_by_t: np.ndarray, lo: float, hi: float, fs: float, order: int = 4) -> np.ndarray:
    """Causal Butterworth band-pass per channel with steady-state initial conditions (Paper A exp_cache)."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    x = x_ch_by_t.T  # samples x channels
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    y, _ = sosfilt(sos, x, axis=0, zi=zi)
    return y.T


def oas_cov(seg_t_by_ch: np.ndarray) -> np.ndarray:
    """OAS-shrunk covariance (Paper A exp_cache.oas_cov, verbatim logic)."""
    x = seg_t_by_ch - seg_t_by_ch.mean(0, keepdims=True)
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    alpha = np.mean(s ** 2)
    num = alpha + mu ** 2
    den = (n + 1.0) * (alpha - mu ** 2 / p)
    shrink = 1.0 if den == 0 else min(num / den, 1.0)
    c = (1.0 - shrink) * s + shrink * mu * np.eye(p)
    floor = max(1e-10 * mu, 1e-12)
    return c + floor * np.eye(p)


# ------------------------------------------------------------------ SPD helpers
def _fun(c, f):
    w, v = np.linalg.eigh(c)
    return (v * f(w)) @ v.T


def sqrtm(c):
    return _fun(c, np.sqrt)


def invsqrtm(c):
    return _fun(c, lambda w: 1.0 / np.sqrt(w))


def logm(c):
    return _fun(c, np.log)


def expm_sym(c):
    return _fun(c, np.exp)


def riemann_mean(covs: np.ndarray, tol: float = 1e-8, maxiter: int = 50) -> np.ndarray:
    """Affine-invariant (Karcher) mean, pyriemann-style gradient descent with adaptive step."""
    m = covs.mean(0)
    nu, tau = 1.0, np.finfo(np.float64).max
    for _ in range(maxiter):
        s, si = sqrtm(m), invsqrtm(m)
        j = np.mean([logm(si @ c @ si) for c in covs], axis=0)
        crit = np.linalg.norm(j, ord="fro")
        h = nu * crit
        m = s @ expm_sym(nu * j) @ s
        if h < tau:
            nu, tau = 0.95 * nu, h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    return m


def upper(t: np.ndarray) -> np.ndarray:
    """Upper triangle with sqrt(2) weights off the diagonal (pyriemann convention)."""
    n = t.shape[-1]
    iu = np.triu_indices(n)
    coeff = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return t[..., iu[0], iu[1]] * coeff


def tangent_space(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    si = invsqrtm(ref)
    return np.stack([upper(logm(si @ c @ si)) for c in covs])


class TSLR:
    """TangentSpace(metric='riemann') + LogisticRegression(C=1.0, max_iter=2000) -- Paper A model M3."""

    def fit(self, covs, y):
        self.ref = riemann_mean(covs)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(tangent_space(covs, self.ref), y)
        return self

    def predict(self, covs):
        return self.clf.predict(tangent_space(covs, self.ref))


# ------------------------------------------------------------------ EEGNet-8,2 (Paper A run_eegnet)
def make_eegnet(C: int, T: int, kern: int = 125, F1: int = 8, D: int = 2, F2: int = 16, p: float = 0.25):
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False), nn.BatchNorm2d(F1))
            self.dw = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)
            self.b2 = nn.Sequential(nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
            self.b3 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                    nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                    nn.AvgPool2d((1, 8)), nn.Dropout(p))
            with torch.no_grad():
                n = self._feat(torch.zeros(1, 1, C, T)).shape[1]
            self.fc = nn.Linear(n, 2)

        def _feat(self, x):
            return self.b3(self.b2(self.dw(self.b1(x)))).flatten(1)

        def forward(self, x):
            return self.fc(self._feat(x))

        def max_norm(self):
            with torch.no_grad():
                self.dw.weight.data = torch.renorm(self.dw.weight.data, 2, 0, 1.0)
                self.fc.weight.data = torch.renorm(self.fc.weight.data, 2, 0, 0.25)

    return EEGNet()


def eegnet_fit_predict(X, y, tr_rows, te_rows, seed, device, epochs=100, bs=64, max_batches=None):
    """X: torch tensor (n,1,C,T) on device (per-trial channel z-scored); y: torch int64 on device."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_eegnet(X.shape[2], X.shape[3]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    tr = torch.as_tensor(np.asarray(tr_rows), device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    done = 0
    for _ in range(epochs):
        perm = tr[torch.randperm(len(tr), generator=gen).to(device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(X[b]), y[b])
            loss.backward()
            opt.step()
            model.max_norm()
            done += 1
            if max_batches is not None and done >= max_batches:
                break
        if max_batches is not None and done >= max_batches:
            break
    model.eval()
    with torch.no_grad():
        te = torch.as_tensor(np.asarray(te_rows), device=device)
        out = model(X[te]).argmax(1)
    return out.cpu().numpy()


# ------------------------------------------------------------------ R4 additions: batched + multiplicity-weighted
def _bfun(c, f):
    w, v = np.linalg.eigh(c)
    return (v * f(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def riemann_mean_w(covs: np.ndarray, w: np.ndarray, tol: float = 1e-8, maxiter: int = 50) -> np.ndarray:
    """Same algorithm as riemann_mean, with sample weights w (= multiplicity of each distinct row).
    With integer weights it equals riemann_mean on the expanded multiset (up to float summation order)."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    m = np.einsum("n,nij->ij", w, covs)
    nu, tau = 1.0, np.finfo(np.float64).max
    for _ in range(maxiter):
        s, si = sqrtm(m), invsqrtm(m)
        j = np.einsum("n,nij->ij", w, _bfun(si @ covs @ si, np.log))
        crit = np.linalg.norm(j, ord="fro")
        h = nu * crit
        m = s @ expm_sym(nu * j) @ s
        if h < tau:
            nu, tau = 0.95 * nu, h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    return m


def tangent_space_b(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    si = invsqrtm(ref)
    return upper(_bfun(si @ covs @ si, np.log))


class TSLRw:
    """TS+LR on a training MULTISET given as distinct rows + integer multiplicities.
    Reference = multiplicity-weighted Riemannian mean (as pyriemann TangentSpace on the expanded rows);
    LogisticRegression(C=1.0, max_iter=2000) is fitted on the EXPANDED rows (each copy is a row)."""

    def fit(self, covs, y, mult):
        mult = np.asarray(mult, dtype=np.int64)
        self.ref = riemann_mean_w(covs, mult)
        feats = tangent_space_b(covs, self.ref)
        rep = np.repeat(np.arange(len(y)), mult)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(feats[rep], np.asarray(y)[rep])
        return self

    def predict(self, covs):
        return self.clf.predict(tangent_space_b(covs, self.ref))


def eegnet_fit_predict_k(X, y, tr_rows, te_rows, seed, device, kern, epochs=100, bs=64, max_batches=None):
    """R3 eegnet_fit_predict with the temporal kernel as an explicit argument (R4: 128 Hz input -> kern=64,
    the original EEGNet-8,2 setting = fs/2, as R3's 125 at 250 Hz). Schedule otherwise identical."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_eegnet(X.shape[2], X.shape[3], kern=kern).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    tr = torch.as_tensor(np.asarray(tr_rows), device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    done = 0
    for _ in range(epochs):
        perm = tr[torch.randperm(len(tr), generator=gen).to(device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(X[b]), y[b])
            loss.backward()
            opt.step()
            model.max_norm()
            done += 1
            if max_batches is not None and done >= max_batches:
                break
        if max_batches is not None and done >= max_batches:
            break
    model.eval()
    with torch.no_grad():
        te = torch.as_tensor(np.asarray(te_rows), device=device)
        out = model(X[te]).argmax(1)
    return out.cpu().numpy()
