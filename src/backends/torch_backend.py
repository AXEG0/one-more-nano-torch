import torch
import torch.nn.functional as F

# Tensor operations
matmul = torch.matmul
add = torch.add
relu = torch.relu
mean = torch.mean

# Loss functions
cross_entropy = F.cross_entropy

# Other operations
set_grad_enabled = torch.set_grad_enabled
