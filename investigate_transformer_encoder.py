import copy
from typing import Iterable, Iterator, Tuple
import torch
import numpy as np

import tocha
from tocha import Tensor
from tocha.module import Parameter
import tocha.nn as nn
import tocha.functional as F
from tocha.nn import TransformerEncoderLayer, Sequential, Linear, Module


def equate_tocha_to_torch_attention(toch, torc, bias, num_heads):
    qkv_weight = torc.in_proj_weight
    out_weight = torc.out_proj.weight
    if bias:
        qkv_bias = torc.in_proj_bias
        out_bias = torc.out_proj.bias

    # separate q,k,v
    q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
    if bias:
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

    # separate head weights/biases
    head_wq = q_weight.t().chunk(dim=1, chunks=num_heads)
    head_wk = k_weight.t().chunk(dim=1, chunks=num_heads)
    head_wv = v_weight.t().chunk(dim=1, chunks=num_heads)
    if bias:
        head_bq = q_bias.chunk(dim=0, chunks=num_heads)
        head_bk = k_bias.chunk(dim=0, chunks=num_heads)
        head_bv = v_bias.chunk(dim=0, chunks=num_heads)

    for h, (wq, wk, wv) in enumerate(zip(head_wq, head_wk, head_wv)):
        head = getattr(toch, f"head_{h}")
        head.q_proj_weight.data = wq.detach().numpy()
        head.k_proj_weight.data = wk.detach().numpy()
        head.v_proj_weight.data = wv.detach().numpy()
    if bias:
        for h, (bq, bk, bv) in enumerate(zip(head_bq, head_bk, head_bv)):
            head = getattr(toch, f"head_{h}")
            head.q_proj_bias.data = bq.detach().numpy()
            head.k_proj_bias.data = bk.detach().numpy()
            head.v_proj_bias.data = bv.detach().numpy()
    toch.out_proj_weight.data = out_weight.t().detach().numpy()
    if bias:
        toch.out_proj_bias.data = out_bias.detach().numpy()


def equate_tocha_to_torch_transformer_encoder_layer(toch, torc):
    equate_tocha_to_torch_attention(toch.self_attn, torc.self_attn, True, toch.nhead)

    toch.linear1.weights.data = torc.linear1.weight.T.detach().numpy()
    toch.linear1.bias.data = torc.linear1.bias.detach().numpy()
    toch.linear2.weights.data = torc.linear2.weight.T.detach().numpy()
    toch.linear2.bias.data = torc.linear2.bias.detach().numpy()
    toch.norm1.weight.data = torc.norm1.weight.detach().numpy()
    toch.norm1.bias.data = torc.norm1.bias.detach().numpy()
    toch.norm2.weight.data = torc.norm2.weight.detach().numpy()
    toch.norm2.bias.data = torc.norm2.bias.detach().numpy()


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

# enc_torch = torch.nn.TransformerEncoder(
#     encoder_layer=enc_layer_torch, num_layers=num_layers
# )

# batch_size = 2
# seq_len = 3

# xnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
# x = torch.tensor(xnp, requires_grad=True)

# out1 = enc_torch(x)
# out2 = enc_layer_torch(enc_layer_torch(x))


# class TransformerEncoder(Module):
#     def __init__(
#         self, encoder_layer: Iterable[TransformerEncoderLayer], num_layers: int
#     ) -> None:
#         super().__init__()
#         self.layers = Sequential([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

#     def forward(self, x: Tensor) -> Tensor:
#         return self.layers(x)

#     # def named_modules(self) -> Iterator[Tuple[str, Parameter]]:
#     #     for n, m in self.layers.named_modules():
#     #         yield n, m
#     # def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
#     #     for n, p in self.layers.named_parameters():
#     #         yield n, p


# enc_layer_tocha = TransformerEncoderLayer(
#     d_model=d_model,
#     dim_feedforwad=dim_feedforward,
#     nhead=n_head,
#     dropout=dropout,
#     layer_norm_eps=layer_norm_eps,
# )
# equate_tocha_to_torch_transformer_encoder_layer(enc_layer_tocha, enc_layer_torch)
# enc_tocha = TransformerEncoder(encoder_layer=enc_layer_tocha, num_layers=num_layers)

# for n, m in enc_torch.named_modules():
#     print(n)
# print("\n")
# for n, m in enc_tocha.named_modules():
#     print(n, type(m))


x = torch.randn(2, 3, 4)
l1 = torch.nn.Linear(4, 5)
l2 = torch.nn.Linear(5, 6)
l3 = torch.nn.Linear(6, 7)
seq = torch.nn.Sequential(l1, l2, l3)

# print(seq[0].weight)
for n, p in seq.named_parameters():
    print(n)
for n, m in seq.named_modules():
    print(n)
print("\n")

l1p = Linear(4, 5)
l2p = Linear(5, 6)
l3p = Linear(6, 7)
seqp = Sequential([l1p, l2p, l3p])
print(seqp[0].weights)