"""R8-D / R8-E model factories with the official hyper-parameters (see vendor_*/UPSTREAM.json).

TCFormer (R8-D, from scratch): configs/tcformer.yaml model_kwargs; Adam lr 9e-4, betas (0.5, 0.999), weight decay 1e-3;
  linear-warmup cosine schedule stepped per epoch (official warmup 20 of 1000 epochs -> 2 of 100 here); input
  (B, 20, 512) at 128 Hz, the same cached 4-40 Hz trials and per-trial channel z-score as C6. No interaug (it synthesises
  training trials by recombining same-class segments, which would change the exposure being measured).
CBraMod (R8-E): pretrained backbone (TUEG) + the official BCIC-IV-2a fine-tuning head 'all_patch_reps'
  (Rearrange -> Linear(C*4*200, 800) -> ELU -> Dropout(0.1) -> Linear(800, 200) -> ELU -> Dropout(0.1) -> Linear(200, 2));
  AdamW, backbone lr 1e-4, head lr 1e-3 * sqrt(64 / 256), weight decay 5e-2, cosine per step to 1e-6, gradient clipping
  1.0, label smoothing 0.1, 50 epochs, batch 64; input (B, 20, 4, 200) = 4 s at 200 Hz in units of 100 uV.
"""

from __future__ import annotations

import math
from pathlib import Path

TCF_KW = dict(F1=32, temp_kernel_lengths=(20, 32, 64), d_group=16, D=2, pool_length_1=8, pool_length_2=7, dropout_conv=0.4,
              use_group_attn=True, q_heads=4, kv_heads=2, trans_depth=5, trans_dropout=0.4, tcn_depth=2, kernel_length_tcn=4,
              dropout_tcn=0.3)
TCF_EPOCHS, TCF_BS, TCF_WARMUP = 100, 64, 2
CBM_EPOCHS, CBM_BS = 50, 64


def make_tcformer(n_channels=20):
    from models.vendor_tcformer.tcformer import TCFormerModule
    return TCFormerModule(n_channels=n_channels, n_classes=2, **TCF_KW)


def tcformer_opt(model, steps_per_epoch, epochs):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=9e-4, betas=(0.5, 0.999), weight_decay=1e-3)

    def fn(ep):  # utils/lr_scheduler.linear_warmup_cosine_decay (per epoch)
        if ep < TCF_WARMUP:
            return float(ep) / float(max(1, TCF_WARMUP))
        return 0.5 * (1.0 + math.cos(math.pi * float(ep - TCF_WARMUP) / float(max(1, epochs - TCF_WARMUP))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, fn), "epoch"


def tcformer_post_step(model):
    return None  # max-norm constraints are applied inside the vendored modules' forward (Conv1dWithConstraint)


def make_cbramod_ft(ckpt: Path, n_channels=20):
    import torch
    from einops.layers.torch import Rearrange
    from torch import nn

    from models.vendor_cbramod.cbramod import CBraMod

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
            self.backbone.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            self.backbone.proj_out = nn.Identity()
            self.classifier = nn.Sequential(Rearrange("b c s d -> b (c s d)"), nn.Linear(n_channels * 4 * 200, 4 * 200), nn.ELU(),
                                            nn.Dropout(0.1), nn.Linear(4 * 200, 200), nn.ELU(), nn.Dropout(0.1), nn.Linear(200, 2))

        def forward(self, x):  # x (B, C, 800) -> (B, C, 4, 200)
            return self.classifier(self.backbone(x.view(x.shape[0], x.shape[1], 4, 200)))
    return Model()


def cbramod_opt(model, steps_per_epoch, epochs):
    import torch
    bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    other = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": 1e-4}, {"params": other, "lr": 1e-3 * (CBM_BS / 256) ** 0.5}], weight_decay=5e-2)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch, eta_min=1e-6), "step"
