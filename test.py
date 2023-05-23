import tocha
import torch
import numpy as np
import tocha.functional as F
import tocha.nn as nn

xnp = np.random.randn(1, 1, 3, 3)
x = tocha.tensor(xnp)
print(x)
conv = nn.Conv2d(1, 1, (2, 2), bias=True)
x = conv(x)
print(x)