"""R15-B model builders: ST-EEGFormer-small, LaBraM-Base, CBraMod (continuity baseline).

Each builder returns a module plus a load report: the checkpoint keys that were missing, unexpected or deliberately
dropped. Architecture-matched random initialisation is the same constructor after torch.manual_seed(seed), with no
checkpoint load. Upstream code is imported from the vendored, hash-verified files; none of it is modified here.

Input conventions (B = batch, C = channels, T = samples):
- ST-EEGFormer: x (B, C, T) at 128 Hz, z-scored per channel and trial; chan_idx from the official 142-name vocabulary;
  patch 16 samples (0.125 s).
- LaBraM: x (B, C, A, 200) at 200 Hz in units of 0.1 mV (uV / 100); input_chans = [0] + 1-based positions in standard_1020;
  patch 200 samples (1 s).
- CBraMod: x (B, C, S, 200) at 200 Hz in units of 0.1 mV; channel identity enters only through channel order (convolutional
  positional encoding); patch 200 samples (1 s).

Checkpoint loading is always torch.load(weights_only=True). The ST-EEGFormer and LaBraM files need an explicit allow-list
(argparse.Namespace for the stored training arguments; for LaBraM also numpy scalar and dtype, codecs.encode). The list was found
by disassembling each pickle, not by trusting the file."""

from __future__ import annotations

import ast
import argparse
import codecs
import importlib.util
import json
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ST_UP = HERE / "vendor_steegformer" / "upstream"
LB_UP = HERE / "vendor_labram" / "upstream"
SRC = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _st_vit():
    return sys.modules.get("r15b_st_vit") or _load("r15b_st_vit", ST_UP / "easy_start__models_vit_eeg.py")


def _st_mae():
    return sys.modules.get("r15b_st_mae") or _load("r15b_st_mae", ST_UP / "pretrain__models_mae_eeg.py")


def _st_lrd():
    return sys.modules.get("r15b_st_lrd") or _load("r15b_st_lrd", ST_UP / "benchmark__neural_networks__util__lr_decay.py")


def _lb(name):
    """LaBraM modules import each other by bare name (modeling_finetune, norm_ema_quantizer); register them under those names."""
    for dep in ("modeling_finetune", "norm_ema_quantizer"):
        if dep not in sys.modules:
            _load(dep, LB_UP / f"{dep}.py")
    return sys.modules.get(name) or _load(name, LB_UP / f"{name}.py")


def st_vocabulary() -> dict:
    return json.loads((HERE / "vendor_steegformer" / "sen_chan_idx.json").read_text())["channels_mapping"]


def st_chan_idx(names) -> list[int]:
    low = {k.lower(): v for k, v in st_vocabulary().items()}
    return [low[c.lower()] for c in names]


def lb_standard_1020() -> list[str]:
    """Channel order extracted verbatim from the pinned upstream feature utilities."""
    return json.loads((LB_UP.parent / "standard_1020.json").read_text())


def lb_input_chans(names) -> list[int]:
    std = lb_standard_1020()
    return [0] + [std.index(c.upper()) + 1 for c in names]


def safe_load(path: Path):
    import numpy.core.multiarray as ncm
    allow = [argparse.Namespace, ncm.scalar, np.dtype, codecs.encode]
    allow += [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
    with torch.serialization.safe_globals(allow):
        return torch.load(path, map_location="cpu", weights_only=True)


def _report(msg, dropped=()):
    return {"missing": sorted(msg.missing_keys), "unexpected": sorted(msg.unexpected_keys), "dropped_before_load": sorted(dropped)}


# ---------------------------------------------------------------- ST-EEGFormer-small
class STClassifier(nn.Module):
    def __init__(self, n_classes, ckpt: Path | None, chan_names, drop_path=0.1):
        super().__init__()
        vit = _st_vit()
        self.net = vit.vit_small_patch16(num_classes=n_classes, drop_path_rate=drop_path, global_pool=True)
        self.register_buffer("chan_idx", torch.tensor(st_chan_idx(chan_names), dtype=torch.long))
        self.load_report = None
        if ckpt is not None:
            sd = safe_load(ckpt)["model"]
            dropped = [k for k in ("head.weight", "head.bias") if k in sd]
            for k in dropped:
                del sd[k]
            self.load_report = _report(self.net.load_state_dict(sd, strict=False), dropped)
            from timm.layers import trunc_normal_
            trunc_normal_(self.net.head.weight, std=2e-5)      # as the official fine-tuning notebook

    def features(self, x):
        with torch.autocast("cuda", enabled=x.is_cuda):
            return self.net.forward_features(x, self.chan_idx.unsqueeze(0).expand(x.shape[0], -1)).float()

    def forward(self, x):
        return self.net(x, self.chan_idx.unsqueeze(0).expand(x.shape[0], -1)).float()   # upstream forward runs under fp16 autocast

    @property
    def head(self):
        return self.net.head

    def param_groups(self, weight_decay=0.05, layer_decay=0.75):
        lrd = _st_lrd()
        nwd = set(self.net.no_weight_decay()) if hasattr(self.net, "no_weight_decay") else set()
        return lrd.param_groups_lrd(self.net, weight_decay=weight_decay, no_weight_decay_list=nwd, layer_decay=layer_decay)


def st_mae(ckpt: Path | None):
    m = _st_mae().mae_vit_small_patch16(norm_pix_loss=False)
    rep = None
    if ckpt is not None:
        sd = safe_load(ckpt)["model"]
        rep = _report(m.load_state_dict(sd, strict=False))
    return m.eval(), rep


# ---------------------------------------------------------------- LaBraM-Base
LB_FT_KW = dict(drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0, use_mean_pooling=True, init_scale=0.001,
                use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)   # README TUAB command + parser defaults


class LBClassifier(nn.Module):
    def __init__(self, n_classes, ckpt: Path | None, chan_names, n_patches):
        super().__init__()
        ft = _lb("modeling_finetune")
        self.net = ft.labram_base_patch200_200(num_classes=n_classes, **LB_FT_KW)
        self.register_buffer("input_chans", torch.tensor(lb_input_chans(chan_names), dtype=torch.long))
        self.n_patches = n_patches
        self.load_report = None
        if ckpt is not None:
            full = safe_load(ckpt)["model"]
            sd = {k[8:]: v for k, v in full.items() if k.startswith("student.")}          # official filter (run_class_finetuning)
            dropped = sorted(k for k in full if not k.startswith("student."))
            st = self.net.state_dict()
            for k in ("head.weight", "head.bias"):
                if k in sd and sd[k].shape != st[k].shape:
                    del sd[k]
                    dropped.append("student." + k)
            for k in [k for k in sd if "relative_position_index" in k]:
                del sd[k]
            self.load_report = _report(self.net.load_state_dict(sd, strict=False), dropped)

    def _shape(self, x):
        return x.view(x.shape[0], x.shape[1], self.n_patches, 200)

    def features(self, x):
        return self.net.forward_features(self._shape(x), input_chans=self.input_chans)

    def forward(self, x):
        return self.net(self._shape(x), input_chans=self.input_chans)

    @property
    def head(self):
        return self.net.head

    def param_groups(self, weight_decay=0.05, layer_decay=0.65):
        of = _lb_optim_factory()
        n = self.net.get_num_layers()
        assigner = of.LayerDecayValueAssigner(list(layer_decay ** (n + 1 - i) for i in range(n + 2)))
        skip = set(self.net.no_weight_decay())
        return of.get_parameter_groups(self.net, weight_decay, skip, assigner.get_layer_id, assigner.get_scale)


def _lb_optim_factory():
    """the three layer-decay definitions of the pinned optim_factory.py, compiled verbatim from its source with ast.
    The module itself cannot be imported: its timm.optim imports (Nadam etc.) no longer exist in the installed timm 1.0.22,
    and no package may be changed. The parameter-group printout is silenced."""
    if "r15b_lb_optim" in sys.modules:
        return sys.modules["r15b_lb_optim"]
    import io
    import contextlib
    import types
    src = (LB_UP / "optim_factory.py").read_text()
    keep = {"get_num_layer_for_vit", "LayerDecayValueAssigner", "get_parameter_groups"}
    body = [n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in keep]
    assert {n.name for n in body} == keep
    mod = types.ModuleType("r15b_lb_optim")
    mod.__dict__.update({"json": json, "torch": torch})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LB_UP / "optim_factory.py"), "exec"), mod.__dict__)
    quiet = mod.get_parameter_groups

    def get_parameter_groups(*a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return quiet(*a, **k)
    mod.get_parameter_groups = get_parameter_groups
    sys.modules["r15b_lb_optim"] = mod
    return mod


def lb_mem(ckpt: Path | None):
    """the masked-EEG-modelling network (student + lm_head) as pretrained; init as run_labram_pretraining (layer scale 0.1)"""
    m = _lb("modeling_pretrain").labram_base_patch200_1600_8k_vocab(drop_path_rate=0.0, init_values=0.1)
    rep = None
    if ckpt is not None:
        rep = _report(m.load_state_dict(safe_load(ckpt)["model"], strict=False))
    return m.eval(), rep


def lb_tokenizer(ckpt: Path | None):
    """the VQ-NSP neural tokenizer (defines the pretraining targets); strict load after the official key filter.
    ckpt=None gives an architecture-matched random tokenizer (no data-dependent k-means codebook initialisation)."""
    vq = _lb("modeling_vqnsp")
    m = vq.vqnsp_encoder_base_decoder_3x200x12(pretrained=False, as_tokenzer=False, n_code=8192, code_dim=64,
                                               **({} if ckpt is not None else {"quantize_kmeans_init": False}))
    if ckpt is None:
        return m.eval(), None
    sd = safe_load(ckpt)["model"]
    dropped = [k for k in sd if k.startswith(("loss", "teacher", "scaling"))]
    for k in dropped:
        del sd[k]
    return m.eval(), _report(m.load_state_dict(sd, strict=True), dropped)


# ---------------------------------------------------------------- CBraMod
def cb_backbone(ckpt: Path | None):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from models.vendor_cbramod.cbramod import CBraMod
    bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    rep = None
    if ckpt is not None:
        rep = _report(bb.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True))
    return bb, rep


class CBClassifier(nn.Module):
    """CBraMod with the head of the official motor-imagery fine-tuning (as R8 / R9-B), sized for S patches of 1 s"""
    def __init__(self, n_classes, ckpt: Path | None, n_channels, n_patches):
        super().__init__()
        self.backbone, self.load_report = cb_backbone(ckpt)
        self.backbone.proj_out = nn.Identity()
        self.n_patches = n_patches
        s = n_patches
        self.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(n_channels * s * 200, s * 200), nn.ELU(), nn.Dropout(0.1),
                                        nn.Linear(s * 200, 200), nn.ELU(), nn.Dropout(0.1), nn.Linear(200, n_classes))

    def _shape(self, x):
        return x.view(x.shape[0], x.shape[1], self.n_patches, 200)

    def features(self, x):
        return self.backbone(self._shape(x)).mean(dim=(1, 2))

    def forward(self, x):
        return self.classifier(self.backbone(self._shape(x)))

    @property
    def head(self):
        return self.classifier

    def param_groups(self, weight_decay=0.05, batch_size=16):
        bb = [p for n, p in self.named_parameters() if n.startswith("backbone.")]
        other = [p for n, p in self.named_parameters() if not n.startswith("backbone.")]
        return [{"params": bb, "lr_scale": 1.0, "weight_decay": weight_decay},
                {"params": other, "lr_scale": 10.0 * (batch_size / 256) ** 0.5, "weight_decay": weight_decay}]   # R8 form: 1e-4 / 1e-3*sqrt(bs/256)


# ---------------------------------------------------------------- registry
SPEC = {
    "ST-EEGFormer-small": {"fs": 128, "patch": 16, "base_lr": 1e-3, "layer_decay": 0.75, "unit": "z-score per channel and trial",
                           "band": (0.1, 75.0), "notch": 50.0},
    "LaBraM-Base": {"fs": 200, "patch": 200, "base_lr": 5e-4, "layer_decay": 0.65, "unit": "uV/100", "band": (0.1, 75.0), "notch": 50.0},
    "CBraMod": {"fs": 200, "patch": 200, "base_lr": 1e-4, "layer_decay": None, "unit": "uV/100", "band": (0.3, 75.0), "notch": 50.0},
}


def build_classifier(name, ckpt, chan_names, n_samples, n_classes=2, seed=0):
    torch.manual_seed(int(seed))
    if name == "ST-EEGFormer-small":
        return STClassifier(n_classes, ckpt, chan_names)
    if name == "LaBraM-Base":
        assert n_samples % 200 == 0, "LaBraM needs whole 1-s patches"
        return LBClassifier(n_classes, ckpt, chan_names, n_samples // 200)
    if name == "CBraMod":
        assert n_samples % 200 == 0, "CBraMod needs whole 1-s patches"
        return CBClassifier(n_classes, ckpt, len(chan_names), n_samples // 200)
    raise KeyError(name)


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def cosine_lr(base, epoch_frac, epochs, warmup, min_lr=1e-6):
    if epoch_frac < warmup:
        return base * epoch_frac / warmup
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch_frac - warmup) / max(epochs - warmup, 1e-9)))


__all__ = ["build_classifier", "st_mae", "lb_mem", "lb_tokenizer", "cb_backbone", "SPEC", "st_chan_idx", "lb_input_chans", "n_params",
           "cosine_lr", "partial"]
