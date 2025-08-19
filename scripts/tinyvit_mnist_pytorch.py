# TinyViT-MNIST (PyTorch) with dataset subsetting for faster debugging
# -------------------------------------------------------------------
# This script loads MNIST, optionally reduces dataset size, trains a minimal
# Vision Transformer, and prints accuracy metrics.

import random
from time import time
import sys
sys.path.append('.')

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.core.tensor import Tensor
from src.core.backend import backend


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
    # Save a batch for later comparison
    # -------------------------------------------------------------------
    X_ref, y_ref = next(iter(train_dl))
    torch.save((X_ref, y_ref), "scripts/batch_ref.pt")

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------

    from src.model import TinyViT

    model = TinyViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_epoch(loader, train=True):
        model.train(train)
        total, correct, loss_total = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with backend.set_grad_enabled(train):
                logits = model(xb)
                loss = backend.cross_entropy(logits, yb)
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
        if epoch == 1:
            model.eval()
            with torch.no_grad():
                logits_ref = model(X_ref.to(DEVICE))
            torch.save(logits_ref, "scripts/logits_ref.pt")
            torch.save(model.state_dict(), "scripts/model_weights.pt")

    print("Done.")


if __name__ == "__main__":
    main()
