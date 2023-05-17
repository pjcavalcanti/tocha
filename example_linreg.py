import numpy as np
import matplotlib.pyplot as plt
from autograd.tensor import Tensor

# True parameters
true_slope = 2.0
true_intercept = -3.0

# Number of data points
num_points = 100

# Generate x values
x_values = np.linspace(-1, 1, num_points)

# Generate y values with added noise
y_values = true_slope * x_values + true_intercept + np.random.normal(0, 0.1, num_points)

x_values = Tensor(x_values)
y_values = Tensor(y_values)


def linear_model(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x * w + b


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    diff = y_true - y_pred
    return (diff * diff).sum() / len(diff)  # type: ignore


# Initialize parameters
slope = Tensor(0.0, requires_grad=True)
intercept = Tensor(0.0, requires_grad=True)

# Define learning rate and number of epochs
learning_rate = 0.1
num_epochs = 100

# Gradient descent loop
for _ in range(num_epochs):
    # Forward pass: compute predicted y
    y_pred = linear_model(x_values, slope, intercept)

    # Compute and print loss
    loss = mean_squared_error(y_values, y_pred)

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    slope.data -= learning_rate * slope.grad.data  # type: ignore
    intercept.data -= learning_rate * intercept.grad.data  # type: ignore

    # Manually zero the gradients after updating weights
    slope.zero_grad()
    intercept.zero_grad()

print(true_intercept, intercept)
print(true_slope, slope)

plt.scatter(x_values.data, y_values.data, label="Data")
plt.plot(
    x_values.data,
    (slope.data * x_values.data + intercept.data),
    label="Fitted line",
    color="red",
)
plt.legend()
plt.show()
