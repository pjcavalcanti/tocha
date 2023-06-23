from unittest import TestCase
import numpy as np

import torch
from tqdm import tqdm

import tocha
from tocha.module import Parameter
from tocha.nn import Module
import tocha.functional as F
from tocha import Tensor
import tocha.optim

class TestAdamAgainstTorch(TestCase):
    def test_mlp_against_torch(self):
        torch.manual_seed(0)
        for _ in range(2):
            N = int(torch.randint(0,50,(1,)))
            noise = torch.rand(1).item()
            n_in = 2
            n_hidden = int(torch.randint(2,6,(1,)))
            n_out = 1
            alpha = 0.01 * torch.rand(1).item()
            beta1 = 0.9 * torch.rand(1).item()
            beta2 = 0.999 * torch.rand(1).item()
            eps = 1e-8 * torch.rand(1).item()
            weight_decay = 0.01 * torch.rand(1).item()
            if torch.rand(1).item() > 0.5:
                weight_decay = 0.0
            epochs = 50
            
            batch_size = 5

            X_torch, Y_torch = generate_data(N, noise)
            X_tocha, Y_tocha = tocha.tensor(X_torch.clone().numpy()), tocha.tensor(Y_torch.clone().numpy())
            
            mlp_torch = MLP_torch(n_in, n_hidden, n_out)
            mlp_tocha = MLP_tocha(n_in, n_hidden, n_out)
            equate_mlp_tocha_torch(mlp_tocha, mlp_torch)
            
            adam_torch = torch.optim.Adam(params=mlp_torch.parameters(), lr=alpha, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
            adam_tocha = tocha.optim.Adam(params=mlp_tocha.parameters(), lr=alpha, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
                
            for e in range(epochs):
                idx = torch.randint(0, X_torch.shape[0], (batch_size,))
                x_torch, y_torch = X_torch[idx], Y_torch[idx].view(-1, 1)
                x_tocha, y_tocha = X_tocha[idx], Y_tocha[idx].reshape((-1, 1))
                
                y_pred_torch = mlp_torch(x_torch)
                y_pred_tocha = mlp_tocha(x_tocha)        
                loss_torch = torch.nn.functional.binary_cross_entropy_with_logits(y_pred_torch, y_torch)
                loss_tocha = F.binary_cross_entropy_with_logits(y_pred_tocha, y_tocha)            
                
                adam_torch.zero_grad()
                loss_torch.backward()
                adam_torch.step()

                adam_tocha.zero_grad()
                loss_tocha.backward()
                adam_tocha.step()
                
                self.assertTrue(np.allclose(loss_torch.detach().numpy(), loss_tocha.data))
                
        
        
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

def equate_mlp_tocha_torch(mlp_toch: MLP_tocha, mlp_torc: MLP_torch):
        mlp_toch.fc1.weights.data = mlp_torc.fc1.weight.clone().T.detach().numpy()
        mlp_toch.fc1.bias.data = mlp_torc.fc1.bias.clone().detach().numpy()
        mlp_toch.fc2.weights.data = mlp_torc.fc2.weight.clone().T.detach().numpy()
        mlp_toch.fc2.bias.data = mlp_torc.fc2.bias.clone().detach().numpy()
