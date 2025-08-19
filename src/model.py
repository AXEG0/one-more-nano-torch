import torch
import torch.nn as nn

from src.core.tensor import Tensor
from src.core.backend import backend

# Model hyperparams
PATCH = 4
DIM = 64
DEPTH = 1
HEADS = 4
MLP_DIM = 128

class TinyViT(nn.Module):
    def __init__(self, patch=PATCH, dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM, num_classes=10):
        super().__init__()
        self.patch_conv = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        num_patches = (28 // patch) ** 2
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_head = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = backend.add(x, self.pos_emb)
        x = self.encoder(x)
        x = backend.mean(x, dim=1)
        return self.cls_head(x)
