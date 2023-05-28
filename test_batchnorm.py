import tocha
import torch
import numpy as np
import tocha.functional as F
import tocha.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)

momentum = 0.1
n_features = 3
length = 100
batch_size = 32
shape = (batch_size, n_features, length)



norm = nn.BatchNorm1d(n_features, momentum=momentum)
norm_torch = torch.nn.BatchNorm1d(n_features,momentum=momentum, dtype=torch.float64)

running_var_error = []
running_mean_error = []
running_var_rel_error = []
running_mean_rel_error = []

for _ in range(1000):
    a_np = np.random.randn(*shape).astype(np.float64)
    a = tocha.tensor(a_np, requires_grad = True)
    a_torch = torch.tensor(a.data, requires_grad = True)

    out = norm(a)
    out_torch = norm_torch(a_torch)

    grad_np = np.random.randn(*out.shape)
    grad = tocha.tensor(grad_np)
    grad_torch = torch.tensor(grad_np)

    out.backward(grad)
    out_torch.backward(grad_torch)

    assert a.grad is not None
    assert a_torch.grad is not None

    # print(f"at iteration {_}")
    # print(f"normalizes correctly: {np.allclose(out.data, out_torch.detach().numpy())}")
    # print(f"backpropagates correctly: {np.allclose(a.grad.data, a_torch.grad.detach().numpy(), atol = 1e-14)}")
    
    error_var = np.abs(norm.running_var.data - norm_torch.running_var.detach().numpy()).sum()
    error_rel_var = error_var / np.abs(norm_torch.running_var.detach().numpy()).sum()
    error_mean = np.abs(norm.running_mean.data - norm_torch.running_mean.detach().numpy()).sum()
    error_rel_mean = error_mean / np.abs(norm_torch.running_mean.detach().numpy()).sum()
    running_var_error.append(error_var)
    running_mean_error.append(error_mean)
    running_var_rel_error.append(error_rel_var)
    running_mean_rel_error.append(error_rel_mean)
    # print("\n\n")

fig, ax = plt.subplots(2,1)
ax[0].plot(running_var_error, label = "var error")
ax[0].plot(running_mean_error, label = "mean error")
ax[1].plot(running_var_rel_error, label = "var rel error")
ax[1].plot(running_mean_rel_error, label = "mean rel error")

ax[0].legend()
ax[1].legend()
plt.show()

norm.eval()
norm_torch.eval()

a_np = np.random.randn(*shape).astype(np.float64)
a = tocha.tensor(a_np, requires_grad = True)
a_torch = torch.tensor(a.data, requires_grad = True)


out2 = norm(a)
out2_torch = norm_torch(a_torch)

print(f"running var \t {norm.running_var.data.reshape(-1)}")
print(f"running var torch {norm_torch.running_var.detach().numpy().reshape(-1)}")

print(f"eval normalizes correctly: {np.allclose(out2.data, out2_torch.detach().numpy())}")
print(f"running mean correct {np.allclose(norm.running_mean.data.reshape(-1), norm_torch.running_mean.detach().numpy().reshape(-1))}") # type: ignore
print(f"running var correct {np.allclose(norm.running_var.data.reshape(-1), norm_torch.running_var.detach().numpy().reshape(-1))}") # type: ignore
print(np.abs((out2.data - out2_torch.detach().numpy())).max())

