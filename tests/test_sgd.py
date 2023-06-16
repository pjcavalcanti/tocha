import unittest

import numpy as np
import torch
import tocha
import tocha.nn
import tocha.optim
import tocha.functional as F

class MLP_tocha(tocha.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = tocha.nn.Linear(input_dim, hidden_dim)
        self.fc2 = tocha.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class MLP_torch(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype=torch.float32):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, dtype=dtype)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim, dtype=dtype)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
def equate_tocha_to_torch_linear(toch, torc):
    toch.weights.data = torc.weight.T.detach().numpy().copy()
    toch.bias.data = torc.bias.detach().numpy().copy()

class TestSGD(unittest.TestCase):
    def test_against_torch_mlp(self):
        # generate circular data with noise
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
        def lr(e):
            if e < 100:
                return 1
            if e < 300:
                return 0.1
            if e < 600:
                return 0.01
            return 0.001
        
        for _ in range(10):
            lr0 = np.random.random() * 10 ** -np.random.randint(1, 5)
            weight_decay = np.random.random() * 10 ** -np.random.randint(1, 5)
            momentum = np.random.random()
            nesterov = bool(np.random.choice([True, False]))
            dampening = 0
            if not nesterov:
                dampening = np.random.random()
            mlp_tocha = MLP_tocha(2, 10, 1)
            mlp_torch = MLP_torch(2, 10, 1, dtype=torch.float64)
            equate_tocha_to_torch_linear(mlp_tocha.fc1, mlp_torch.fc1)
            equate_tocha_to_torch_linear(mlp_tocha.fc2, mlp_torch.fc2)
            optimizer_tocha = tocha.optim.SGD(mlp_tocha.parameters(), lr=lr0, momentum=momentum, nesterov=nesterov, dampening=dampening, weight_decay=weight_decay)
            optimizer_torch = torch.optim.SGD(mlp_torch.parameters(),lr=lr0,momentum=momentum, nesterov=nesterov, dampening=dampening, weight_decay=weight_decay)
            
            for epoch in range(40):
                idx = np.random.randint(0, len(X), (32,))

                x_tocha = X_tocha[idx]
                y_tocha = Y_tocha[idx]
                y_pred_tocha = mlp_tocha(x_tocha)
                loss_tocha = ((y_pred_tocha - y_tocha) * (y_pred_tocha - y_tocha)).sum()

                x_torch = X_torch[idx]
                y_torch = Y_torch[idx]
                y_pred_torch = mlp_torch(x_torch)
                loss_torch = ((y_pred_torch - y_torch) * (y_pred_torch - y_torch)).sum()

                optimizer_tocha.zero_grad()
                loss_tocha.backward()
                optimizer_tocha.step()
                optimizer_tocha.lr = lr(epoch)  # Update the learning rate

                optimizer_torch.zero_grad()
                loss_torch.backward()
                optimizer_torch.step()
                for param_group in optimizer_torch.param_groups:
                    param_group['lr'] = lr(epoch)  # Update the learning rate

                passloss = np.isclose(loss_torch.item(), loss_tocha.data, atol=1e-5)
                passgrad1 = np.allclose(mlp_tocha.fc1.weights.grad.data, mlp_torch.fc1.weight.grad.T.data, atol=1e-5) # type: ignore
                passgrad2 = np.allclose(mlp_tocha.fc2.weights.grad.data, mlp_torch.fc2.weight.grad.T.data, atol=1e-5) # type: ignore
                passgrad3 = np.allclose(mlp_tocha.fc1.bias.grad.data, mlp_torch.fc1.bias.grad.data, atol=1e-5) # type: ignore
                passgrad4 = np.allclose(mlp_tocha.fc2.bias.grad.data, mlp_torch.fc2.bias.grad.data, atol=1e-5) # type: ignore
                assert passloss, "losses not close"
                assert passgrad1, "grads weights layer1 not close"
                assert passgrad2, "grads weights layer2 not close"
                assert passgrad3, "grads bias layer1 not close"
                assert passgrad4, "grads bias layer2 not close"