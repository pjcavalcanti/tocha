import tocha
import torch
import numpy as np
import tocha.functional as F
import tocha.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)

batch_size = np.random.randint(2, 32)
n_features = np.random.randint(1, 10)
height = np.random.randint(1, 50)
width = np.random.randint(1,50)
dtype = torch.float64
device = torch.device("cpu")

momentum = np.random.random()
eps = np.random.random() * 10 ** (-np.random.randint(1, 10))
affine=True
track_running_stats=True

shape = (batch_size, n_features, height, width)
x = tocha.tensor(np.random.randn(*shape).astype(np.float64),requires_grad=True) # type: ignore
x_torch = torch.tensor(x.data, requires_grad=True, device=device, dtype=dtype)

norm = nn.BatchNorm2d(n_features, eps=eps, momentum=momentum)
norm_torch = torch.nn.BatchNorm2d(n_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

out = norm(x)
out_torch = norm_torch(x_torch)

print(x.shape)
print(out.shape)
print(out_torch.shape)

grad = tocha.tensor(np.random.randn(*out.shape))
grad_torch = torch.tensor(grad.data, device=device, dtype=dtype)

out.backward(grad)
out_torch.backward(grad_torch)

print(norm.running_mean.data.squeeze())
print(norm_torch.running_mean.detach().numpy())
print("\n\n")
print(norm.running_var.data.squeeze())
print(norm_torch.running_var.detach().numpy())
print("\n")