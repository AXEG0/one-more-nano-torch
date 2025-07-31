import torch
import sys
sys.path.append('.')

from src.core.backend import backend
from src.model import TinyViT

def main():
    # Load the reference batch and logits
    X_ref, y_ref = torch.load("scripts/batch_ref.pt")
    logits_ref = torch.load("scripts/logits_ref.pt")

    # Run inference with the torch backend
    backend.set_backend("torch")
    model = TinyViT()
    model.load_state_dict(torch.load("scripts/model_weights.pt"))
    model.eval()
    # We need to load the weights from the trained model
    # For now, let's just do a forward pass with random weights
    logits_torch = model(X_ref)

    # Run inference with the python backend
    backend.set_backend("python")
    # This will fail because the python backend doesn't support all the operations yet
    # logits_python = model(X_ref)

    # Compare the results
    print("Torch logits:", logits_torch)
    # print("Python logits:", logits_python)
    print("Reference logits:", logits_ref)

if __name__ == "__main__":
    main()
