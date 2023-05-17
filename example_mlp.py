from tocha.nn import Linear, Module, Parameter
from autograd.tensor import Tensor
from tocha.functional import relu, sigmoid
import matplotlib.pyplot as plt
import numpy as np

# define MLP
class MLP(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = sigmoid(self.fc2(x))
        return x

# generate 2D data
np.random.seed(0)
num_samples = 100
X = np.random.rand(num_samples, 2)
Y = np.zeros((num_samples, 1))
Y[X[:, 0] > X[:, 1]] = 1

# add Gaussian noise
X += np.random.normal(0, 0.1, size=X.shape)

# Train
mlp = MLP(2, 10, 1)
for epoch in range(100):
    for x, y in zip(X, Y):
        y_pred = mlp(Tensor(x))
        loss = (y_pred - Tensor(y))**2
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.1 * p.grad.data
            p.zero_grad()

# plot decision boundary
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
X1, X2 = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in
