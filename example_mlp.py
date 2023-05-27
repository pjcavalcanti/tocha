import tocha
import tocha.nn as nn
from tocha.functional import relu, sigmoid
import matplotlib.pyplot as plt
import numpy as np


# define MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = sigmoid(self.fc1(x))
        x = sigmoid(self.fc2(x))
        return x


# generate circular data
np.random.seed(0)
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
X = np.vstack([X_inner, X_outer])
Y = np.vstack([Y_inner, Y_outer])

# add Gaussian noise
X += np.random.normal(0, 0.3, size=X.shape)

# go to tensors
Xt = tocha.tensor(X)
Yt = tocha.tensor(Y)

# Train
mlp = MLP(2, 10, 1)


def lr(e):
    if e < 100:
        return 1
    if e < 300:
        return 0.1
    if e < 600:
        return 0.01
    return 0.001


for epoch in range(2000):
    idx = np.random.randint(0, len(X), (3))
    x = Xt[idx]
    y = Yt[idx]
    y_pred = mlp(x)
    loss = ((y_pred - y) * (y_pred - y)).sum()
    print(loss)
    loss.backward()
    # for p in mlp.parameters():
    for p in mlp.parameters():
        p.data -= lr(epoch) * p.grad.data  # type: ignore
        p.zero_grad()


# plot decision boundary
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
X1, X2 = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = mlp(tocha.tensor(np.array([X1[i, j], X2[i, j]]))).data

plt.contourf(X1, X2, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], edgecolors="k")
plt.show()
