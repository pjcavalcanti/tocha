import tocha
import torch
import numpy as np
import tocha.functional as F

dims = (2, 3, 4)
a_np = np.array([i+1 for i in range(np.prod(dims))]).reshape(dims).astype(np.float32)
a = tocha.tensor(a_np, requires_grad=True)
a_torch = torch.tensor(a_np, requires_grad=True)

ax = (1)
b = a.sum(ax)
b_toch = a_torch.sum(ax)
print(b)
print(b_toch)

b_toch.backward(torch.ones_like(b_toch))