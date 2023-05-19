import tocha
import tocha.nn as nn
from tocha.functional import relu, sigmoid, flatten, tanh
import matplotlib.pyplot as plt
import numpy as np
from autograd.tensor import Tensor


# Let's create a simple sinusoidal function as our data
np.random.seed(0)

# Create a sequence data
x = np.linspace(0, 10, 500)
y = np.sin(x) + np.random.normal(0, 0.1, len(x))

# Normalize input to (0, 1) range
x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())

# Reshape to (N, C, L) format, where N is batch size, C is number of channels, and L is length
x = x.reshape(1, 1, -1)
y = y.reshape(1, 1, -1)


class SimpleConv1dModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * (len(x[0][0]) - 5 + 1), out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = relu(x)
        x = flatten(x)  # Flatten
        x = self.fc1(x)
        return x


# Hyperparameters
learning_rate = 0.01
num_epochs = 100

# Instantiate the model
model = SimpleConv1dModel()


# Loss function
def loss_fn(y_pred: Tensor, y: Tensor) -> Tensor:
    return ((y_pred - y) * (y_pred - y)).sum()


# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(Tensor(x))

    # Compute loss
    loss = loss_fn(y_pred, Tensor(y))

    # Zero gradients
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data  # type: ignore

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.data}")

# Plot actual vs predicted values
y_pred = model(Tensor(x)).data.reshape(-1)
plt.plot(x[0, 0], y[0, 0], label="Actual")
plt.plot(x[0, 0], y_pred, label="Predicted")
plt.legend()
plt.show()
