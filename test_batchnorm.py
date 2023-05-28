import tocha
import torch
import numpy as np
import tocha.functional as F
import tocha.nn as nn


n_features = 2
batch_size = 32

a_np = np.random.randn(batch_size, n_features).astype(np.float64)
a = tocha.tensor(a_np, requires_grad = True)
a_torch = torch.tensor(a.data, requires_grad = True)


norm = nn.BatchNorm1d(n_features)
norm_torch = torch.nn.BatchNorm1d(n_features, dtype=torch.float64)

out = norm(a)
out_torch = norm_torch(a_torch)


print(f"normalizes correctly: {np.allclose(out.data, out_torch.detach().numpy())}")

grad_np = np.random.randn(*out.shape)
grad = tocha.tensor(grad_np)
grad_torch = torch.tensor(grad_np)

out.backward(grad)
out_torch.backward(grad_torch)

assert a.grad is not None
assert a_torch.grad is not None

print(f"backpropagates correctly: {np.allclose(a.grad.data, a_torch.grad.detach().numpy(), atol = 1e-14)}")

# norm.eval()
# norm_torch.eval()

# a_np = np.random.randn(batch_size, n_features).astype(np.float64)
# a = tocha.tensor(a_np, requires_grad = True)
# a_torch = torch.tensor(a.data, requires_grad = True)


# out2 = norm(a)
# out2_torch = norm_torch(a_torch)


# print(f"running mean correct {np.allclose(norm.running_mean.data, norm_torch.running_mean.detach().numpy())}")

