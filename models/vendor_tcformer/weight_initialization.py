# Vendored from https://github.com/altaheri/TCFormer @ 701a36164070 (utils/weight_initialization.py); MIT licence, see LICENSE in this directory. No edits.
from torch import nn


def glorot_weight_zero_bias(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            if "norm" not in module.__class__.__name__.lower():
                nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
