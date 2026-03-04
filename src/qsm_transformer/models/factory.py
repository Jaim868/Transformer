from __future__ import annotations

from torch import nn

from .resnet_qsm import ResNetQSM
from .transformer_qsm import QSMPatchTransformer
from .unet import UNet2D


def build_model(name: str, **kwargs) -> nn.Module:
    registry = {
        "unet": UNet2D,
        "resnet": ResNetQSM,
        "transformer": QSMPatchTransformer,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
