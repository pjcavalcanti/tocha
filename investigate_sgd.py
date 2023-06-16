from typing import Iterable, Union
import torch

import tocha
from tocha.module import Parameter
import tocha.nn as nn
from tocha import Tensor
from tocha.functional import relu, sigmoid
from tocha.optim import Optimizer

import matplotlib.pyplot as plt
import numpy as np


def equate_tocha_to_torch_linear(toch, torc):
    toch.weights.data = torc.weight.T.detach().numpy().copy()
    toch.bias.data = torc.bias.detach().numpy().copy()


# define MLP
class MLP_tocha(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = sigmoid(self.fc1(x))
        x = sigmoid(self.fc2(x))
        # x = self.fc1(x)
        # x = self.fc2(x)
        return x


class MLP_torch(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype=torch.float32):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, dtype=dtype)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim, dtype=dtype)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # x = self.fc1(x)
        # x = self.fc2(x)
        return x


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        momentum: float = 0,
        nesterov: bool = False,
    ) -> None:
        assert lr > 0
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        self.lr = lr
        self.params = list(params)
        self.v = [np.zeros(p.shape).astype(p.data.dtype) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            # In Sutskever et al. we have differently, but I will mirror pytorch
            self.v[i] = self.momentum * self.v[i] + p.grad.data
            p.data = p.data - self.lr * self.v[i]


# generate circular data
np.random.seed(0)
torch.manual_seed(0)
num_samples = 500

# Inner circle
inner_radius = 2
inner_angle = np.random.uniform(0, 2 * np.pi, num_samples // 2)
inner_x = inner_radius * np.cos(inner_angle)
inner_y = inner_radius * np.sin(inner_angle)
X_inner = np.column_stack([inner_x, inner_y])
Y_inner = np.zeros((num_samples // 2, 1))

# Outer circle
outer_radius = 5
outer_angle = np.random.uniform(0, 2 * np.pi, num_samples // 2)
outer_x = outer_radius * np.cos(outer_angle)
outer_y = outer_radius * np.sin(outer_angle)
X_outer = np.column_stack([outer_x, outer_y])
Y_outer = np.ones((num_samples // 2, 1))

# Concatenate inner and outer circle
X = np.vstack([X_inner, X_outer]).astype(np.float64)
Y = np.vstack([Y_inner, Y_outer]).astype(np.float64)

# add Gaussian noise
X += np.random.normal(0, 0.3, size=X.shape)

X = X.astype(np.float64)
Y = Y.astype(np.float64)

# go to tensors
X_tocha = tocha.tensor(X)
Y_tocha = tocha.tensor(Y)
X_torch = torch.tensor(X)
Y_torch = torch.tensor(Y)

# Train
lr = 0.1
momentum = 0.9
mlp_tocha = MLP_tocha(2, 10, 1)
mlp_torch = MLP_torch(2, 10, 1, dtype=torch.float64)
equate_tocha_to_torch_linear(mlp_tocha.fc1, mlp_torch.fc1)
equate_tocha_to_torch_linear(mlp_tocha.fc2, mlp_torch.fc2)
optimizer_tocha = SGD(mlp_tocha.parameters(), lr=lr, momentum=momentum)
optimizer_torch = torch.optim.SGD(mlp_torch.parameters(),lr=lr,momentum=momentum)

out1 = mlp_tocha(X_tocha)
out2 = mlp_torch(X_torch)
# print(np.allclose(out1.data, out2.detach().numpy()))
print(type(mlp_tocha), type(mlp_torch))
print(type(optimizer_tocha), type(optimizer_torch))
print(type(X_tocha.data.dtype), type(X_torch.data.dtype))

# def lr(e):
#     if e < 100:
#         return 1
#     if e < 300:
#         return 0.1
#     if e < 600:
#         return 0.01
#     return 0.001


losses_tocha = []
losses_torch = []

for p in mlp_tocha.parameters():
    print(p.shape)
for p in mlp_torch.parameters():
    print(p.shape)
print(f"{optimizer_tocha.lr=}")
print(f"{optimizer_torch.param_groups[0]['lr']=}")

for epoch in range(300):
    idx = np.random.randint(0, len(X), (32,))

    x_tocha = X_tocha[idx]
    y_tocha = Y_tocha[idx]
    y_pred_tocha = mlp_tocha(x_tocha)
    loss_tocha = ((y_pred_tocha - y_tocha) * (y_pred_tocha - y_tocha)).sum()
    losses_tocha.append(loss_tocha.data)

    x_torch = X_torch[idx]
    y_torch = Y_torch[idx]
    y_pred_torch = mlp_torch(x_torch)
    loss_torch = ((y_pred_torch - y_torch) * (y_pred_torch - y_torch)).sum()
    losses_torch.append(loss_torch.item())

    optimizer_tocha.zero_grad()
    loss_tocha.backward()
    optimizer_tocha.step()

    optimizer_torch.zero_grad()
    loss_torch.backward()
    optimizer_torch.step()

    # print(loss_tocha.data, loss_torch.item())
    passloss = np.isclose(loss_torch.item(), loss_tocha.data, atol=1e-5)
    passgrad1 = np.allclose(mlp_tocha.fc1.weights.grad.data, mlp_torch.fc1.weight.grad.T.data, atol=1e-5)
    passgrad2 = np.allclose(mlp_tocha.fc2.weights.grad.data, mlp_torch.fc2.weight.grad.T.data, atol=1e-5)
    passgrad3 = np.allclose(mlp_tocha.fc1.bias.grad.data, mlp_torch.fc1.bias.grad.data, atol=1e-5)
    passgrad4 = np.allclose(mlp_tocha.fc2.bias.grad.data, mlp_torch.fc2.bias.grad.data, atol=1e-5)
    print(passloss, passgrad1, passgrad2, passgrad3, passgrad4)

    print("===============")            

# plot decision boundary
# x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
# y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
# X1, X2 = np.meshgrid(x_range, y_range)
# Z = np.zeros_like(X1)
# for i in range(X1.shape[0]):
#     for j in range(X1.shape[1]):
#         Z[i, j] = mlp(tocha.tensor(np.array([X1[i, j], X2[i, j]]))).data

# plt.contourf(X1, X2, Z, alpha=0.4)
# plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], edgecolors="k")
# plt.show()

plt.cla()
plt.plot(losses_tocha, label="tocha")
plt.plot(losses_torch, label="torch")
plt.legend()
plt.show()
