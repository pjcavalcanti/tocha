import torch
import tocha
from tocha import tensor
import numpy as np


anp = np.array([1, 2, 3, 4, 5]).astype(np.float32)
a = tensor(anp, requires_grad=True)
a_torch = torch.tensor(anp, requires_grad=True)

b = a
b[3] = 0
b_torch = a_torch
b_torch[3] = 0

gradnp = np.random.randn(*b.shape)
grad = tocha.tensor(gradnp)
grad_torch = torch.tensor(gradnp)

b.backward(grad)
b_torch.backward(grad_torch)

print(a.grad, a_torch.grad)
