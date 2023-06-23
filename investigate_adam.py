import torch
from torch.optim.optimizer import Optimizer

import tocha
from tocha.nn import Module
import tocha.functional as F
from tocha import Tensor
from tocha.optim import Optimizer

from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_points(N, r, noise):
    r = torch.abs(torch.randn(N)* noise + r)
    theta = torch.randn(N) * 2 * torch.pi
    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    return x
def generate_data(N, noise=0.1):
    x1 = generate_points(N, 1,noise)
    x2 = generate_points(N, 2,noise)
    y1 = torch.zeros(N)
    y2 = torch.ones(N)
    x = torch.cat([x1, x2], dim=0)
    y = torch.cat([y1, y2], dim=0)
    rd = torch.randperm(2*N)
    return x[rd], y[rd]

def plot_decision_boundary(model, X, Y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h))
    grid_tensor = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], dim=-1)

    # Predict the function value for the whole grid
    with torch.no_grad():
        model.eval()
        Z = torch.sigmoid(model(grid_tensor.view(-1, 2))).view(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.show()

class MLP_torch(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
        self.fc1 = torch.nn.Linear(n_in, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_out)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_tocha(tocha.nn.Module):
    def __init__(self, n_in:int, n_hidden:int, n_out: int) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
        self.fc1 = tocha.nn.Linear(n_in, n_hidden)
        self.fc2 = tocha.nn.Linear(n_hidden, n_out)
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class Adam_torch():
    def __init__(self, params, lr, betas, eps, weight_decay=None) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay if (weight_decay is not None and weight_decay != 0) else None
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
    
    def step(self):
        self.t = self.t + 1
        if self.weight_decay is None:
            for i, p in enumerate(self.params):
                print("a")
                g_t = p.grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g_t**2
                p.data += - self.lr * (self.m[i] / (1 - self.beta1 ** self.t)) / (torch.sqrt(self.v[i] / (1 - self.beta2 ** self.t)) + self.eps)
        else:
            for i, p in enumerate(self.params):
                g_t = p.grad + self.weight_decay * p.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g_t**2
                p.data += - self.lr * (self.m[i] / (1 - self.beta1 ** self.t)) / (torch.sqrt(self.v[i] / (1 - self.beta2 ** self.t)) + self.eps)

torch.manual_seed(0)   
    
def adam_torch_manual():
    N = 100
    noise = 0.2
    n_in = 2
    n_hidden = 5
    n_out = 1
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.1

    epochs = 1000
    batch_size = 5

    X, Y = generate_data(N, noise)
    mlp_torch1 = MLP_torch(n_in, n_hidden, n_out)
    mlp_torch2 = MLP_torch(n_in, n_hidden, n_out)
    mlp_torch2.fc1.weight = torch.nn.Parameter(mlp_torch1.fc1.weight.clone())
    mlp_torch2.fc1.bias = torch.nn.Parameter(mlp_torch1.fc1.bias.clone())
    mlp_torch2.fc2.weight = torch.nn.Parameter(mlp_torch1.fc2.weight.clone())
    mlp_torch2.fc2.bias = torch.nn.Parameter(mlp_torch1.fc2.bias.clone())

    print(mlp_torch1(X[:5]), mlp_torch2(X[:5]))

    optimizer1 = torch.optim.Adam(params=mlp_torch1.parameters(), lr=alpha, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    optimizer2 = Adam_torch(params=mlp_torch2.parameters(), lr=alpha, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    losses = []
    for e in tqdm(range(epochs)):
        idx = torch.randint(0, X.shape[0], (batch_size,))
        x, y = X[idx], Y[idx].view(-1, 1)
        y_pred1 = mlp_torch1(x)
        y_pred2 = mlp_torch2(x)
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(y_pred1, y)
        loss2 = torch.nn.functional.binary_cross_entropy_with_logits(y_pred2, y)
        losses.append(loss1.item())
        
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        print(torch.allclose(loss1, loss2))
    losses.append(loss1.item())

    plot_decision_boundary(mlp_torch1, X, Y)
    
def adam_tocha_vs_torch():
    
    def equate_mlp_tocha_torch(mlp_toch: MLP_tocha, mlp_torc: MLP_torch):
            mlp_toch.fc1.weights.data = mlp_torc.fc1.weight.clone().T.detach().numpy()
            mlp_toch.fc1.bias.data = mlp_torc.fc1.bias.clone().detach().numpy()
            mlp_toch.fc2.weights.data = mlp_torc.fc2.weight.clone().T.detach().numpy()
            mlp_toch.fc2.bias.data = mlp_torc.fc2.bias.clone().detach().numpy()
            
    N = 100
    noise = 0.2
    n_in = 2
    n_hidden = 5
    n_out = 1
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.1

    epochs = 5
    batch_size = 5

    X_torch, Y_torch = generate_data(N, noise)
    X_tocha, Y_tocha = tocha.tensor(X_torch.clone().numpy()), tocha.tensor(Y_torch.clone().numpy())
    
    # print(np.allclose(X_torch.detach().numpy(), X_tocha.data))
    # print(np.allclose(Y_torch.detach().numpy(), Y_tocha.data))
    
    mlp_torch = MLP_torch(n_in, n_hidden, n_out)
    mlp_tocha = MLP_tocha(n_in, n_hidden, n_out)
    equate_mlp_tocha_torch(mlp_tocha, mlp_torch)
    
    # print(mlp_torch(X_torch[:5]), mlp_tocha(X_tocha[:5]))
    
    adam_torch = torch.optim.Adam(params=mlp_torch.parameters(), lr=alpha, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    # optimizer2 = Adam_torch(params=mlp_tocha.parameters(), lr=alpha, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    losses = []
    for e in tqdm(range(epochs)):
        idx = np.random.randint(0, X_torch.shape[0], (batch_size,))
        x_torch, y_torch = X_torch[idx], Y_torch[idx].view(-1, 1)
        x_tocha, y_tocha = X_tocha[idx], Y_tocha[idx].reshape((-1, 1))
        
        y_pred_torch = mlp_torch(x_torch)
        y_pred_tocha = mlp_tocha(x_tocha)
        loss_torch = torch.nn.functional.binary_cross_entropy_with_logits(y_pred_torch, y_torch)
        loss_tocha = F.binary_cross_entropy_with_logits(y_pred_tocha, y_tocha)
 
        print(loss_torch.item(), loss_tocha.data)
        
        
        # adam_torch.zero_grad()
        # loss1.backward()
        # adam_torch.step()
        
        # optimizer2.zero_grad()
        # loss2.backward()
        # optimizer2.step()

        # print(torch.allclose(loss_torch, loss_tocha))
    losses.append(loss_torch.item())

adam_tocha_vs_torch()