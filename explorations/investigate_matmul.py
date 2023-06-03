import torch

x = torch.randn(2,3,4,8,6)
y = torch.randn(2,3,4,6,5)

lenx = len(x.shape)
leny = len(y.shape)

zd = x @ y
# zt = torch.zeros_like(zd)
# zt = torch.tensordot(x, y, dims=([lenx-1],[leny-2]))
# print(zd == zt)
print(zd.shape)