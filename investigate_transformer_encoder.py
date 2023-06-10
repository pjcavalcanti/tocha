import torch
import numpy as np

import tocha
from tocha import Tensor
import tocha.nn as nn
import tocha.functional as F
from tocha.nn import TransformerEncoderLayer, Sequential, Linear

# n_head = 3
# d_model = 5 * n_head
# dim_feedforward = 7
# dropout = 0.0
# layer_norm_eps = 1e-5

# num_layers = 2

# enc_layer_torch = torch.nn.TransformerEncoderLayer(
#     d_model=d_model,
#     nhead=n_head,
#     dim_feedforward=dim_feedforward,
#     dropout=dropout,
#     layer_norm_eps=layer_norm_eps,
#     batch_first=True,
# )

# enc_torch = torch.nn.TransformerEncoder(enc_layer_torch, num_layers=num_layers)

# batch_size = 2
# seq_len = 3

# xnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
# x = torch.tensor(xnp, requires_grad=True)

# out1 = enc_torch(x)
# out2 = enc_layer_torch(enc_layer_torch(x))

# print(torch.allclose(out1, out2))

x = torch.randn(2, 3, 4)
l1 = torch.nn.Linear(4, 5)
l2 = torch.nn.Linear(5, 6)
l3 = torch.nn.Linear(6, 7)
seq = torch.nn.Sequential(l1, l2, l3)

# print(seq[0].weight)
for n, m in seq.named_modules():
    print(n)
print("\n")

l1p = Linear(4, 5)
l2p = Linear(5, 6)
l3p = Linear(6, 7)
seqp = Sequential([l1p, l2p, l3p])

for n, m in seqp.named_modules():
    print(n)

for n, p in seqp.named_parameters():
    print(n)