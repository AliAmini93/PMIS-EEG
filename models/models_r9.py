"""R9-B CBraMod variants. Identical to R8's pmis_r8.models_de.make_cbramod_ft (same construction order, so the classifier
head draws the same random initialisation for the same seed) except for (a) whether the TUEG checkpoint is loaded and
(b) the number of input channels (head width)."""

from __future__ import annotations

from pathlib import Path


def make_cbramod(ckpt: Path | None, n_channels: int = 20):
    import torch
    from einops.layers.torch import Rearrange
    from torch import nn

    from models.vendor_cbramod.cbramod import CBraMod

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
            if ckpt is not None:
                self.backbone.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            self.backbone.proj_out = nn.Identity()
            self.classifier = nn.Sequential(Rearrange("b c s d -> b (c s d)"), nn.Linear(n_channels * 4 * 200, 4 * 200), nn.ELU(),
                                            nn.Dropout(0.1), nn.Linear(4 * 200, 200), nn.ELU(), nn.Dropout(0.1), nn.Linear(200, 2))

        def forward(self, x):  # x (B, C, 800) -> (B, C, 4, 200)
            return self.classifier(self.backbone(x.view(x.shape[0], x.shape[1], 4, 200)))
    return Model()


def make_backbone(ckpt: Path | None, seed: int, keep_head: bool):
    """Backbone for reconstruction (keep_head=True keeps the pretraining proj_out) or for embeddings (proj_out = Identity).
    Random variants are constructed after torch.manual_seed(seed); the pretrained variant is constructed the same way and
    then overwritten by the checkpoint (strict load: every key must match)."""
    import torch

    from models.vendor_cbramod.cbramod import CBraMod
    torch.manual_seed(int(seed))
    bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    if ckpt is not None:
        bb.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    if not keep_head:
        bb.proj_out = torch.nn.Identity()
    return bb.eval()
