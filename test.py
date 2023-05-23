import tocha
import torch
import numpy as np
import tocha.functional as F

x = torch.randn(2,2,2)
print(x, x.shape)
expx = torch.exp(x)
print(expx, expx.shape)
dim = (1,2)
sft = expx / expx.sum(dim=dim, keepdim=True)
print(sft, sft.shape)
print(expx.sum(dim=dim, keepdim=True), expx.sum(dim=dim, keepdim=True).shape)