from __future__ import annotations

import torch
from torch import nn


class QSMPatchTransformer(nn.Module):
    """ViT-style patch transformer for 2D QSM reconstruction."""

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(b, -1, c * p * p)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, patch_dim = patches.shape
        p = self.patch_size
        h = w = self.image_size // p
        x = patches.view(b, h, w, 1, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(b, 1, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(self._patchify(x)) + self.pos_embed
        encoded = self.encoder(tokens)
        patches = self.decoder(encoded)
        return self._unpatchify(patches)
