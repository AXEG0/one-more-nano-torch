import torch

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = torch.tensor(data, requires_grad=requires_grad)
        self.grad = None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
