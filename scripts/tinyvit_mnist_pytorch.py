# TinyViT-MNIST (PyTorch) with dataset subsetting for faster debugging
# -------------------------------------------------------------------
# This script loads MNIST, optionally reduces dataset size, trains a minimal
# Vision Transformer, and prints accuracy metrics.

import random
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def set_seed(seed: int) -> None:
    """Helper to ensure reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Subset sizes (set to None or 0 to use full dataset)
TRAIN_SUBSET = 10_000
TEST_SUBSET = 1_000

# Model hyperparams
PATCH = 4
DIM = 64
DEPTH = 1
HEADS = 4
MLP_DIM = 128
EPOCHS = 5
BS = 256
LR = 3e-4
WEIGHT_DECAY = 0.01


set_seed(SEED)

# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_full = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=tfm)
test_full = torchvision.datasets.MNIST(root=".", train=False, download=True, transform=tfm)


def make_subset(dataset, n, seed=SEED):
    if (n is None) or (n <= 0) or (n >= len(dataset)):
        return dataset
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    idxs = perm[:n]
    return Subset(dataset, idxs)


def main() -> None:
    train_ds = make_subset(train_full, TRAIN_SUBSET)
    test_ds = make_subset(test_full, TEST_SUBSET)

    print(f"Train samples: {len(train_ds)} / {len(train_full)} total")
    print(f"Test samples : {len(test_ds)} / {len(test_full)} total")

    train_dl = DataLoader(train_ds, BS, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, BS, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------

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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patch_conv(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_emb
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.cls_head(x)

    model = TinyViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_epoch(loader, train=True):
        model.train(train)
        total, correct, loss_total = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.set_grad_enabled(train):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            total += yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            loss_total += loss.item() * yb.size(0)
        return loss_total / total, correct / total

    for epoch in range(1, EPOCHS + 1):
        t0 = time()
        tr_loss, tr_acc = run_epoch(train_dl, train=True)
        te_loss, te_acc = run_epoch(test_dl, train=False)
        dt = time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train {tr_acc*100:5.2f}% (loss {tr_loss:.3f}) | "
            f"test {te_acc*100:5.2f}% (loss {te_loss:.3f}) | "
            f"{dt:.1f}s",
        )

    print("Done.")


if __name__ == "__main__":
    main()
