import tocha
import torch
import numpy as np
import tocha.functional as F
import tocha.nn as nn

xnp = np.random.randn(1, 1, 3, 3)
x = tocha.tensor(xnp)
x_torch = torch.tensor(xnp)

m = x.mean()
m_torch = x_torch.mean()

print(m.data == m_torch.item())