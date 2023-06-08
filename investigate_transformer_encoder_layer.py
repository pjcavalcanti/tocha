import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import tocha
from tocha import Tensor
import tocha.functional as F
import tocha.nn as nn
from tocha.nn import Module, LayerNorm, Dropout, Linear, MultiheadAttention


def making_torch_manually():
    class TransformerEncoderLayer(torch.nn.Module):
        def __init__(
            self, d_model, nhead, dim_feedforwad, dropout, layer_norm_eps=1e-5
        ):
            super().__init__()
            self.self_attn = torch.nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.linear1 = torch.nn.Linear(d_model, dim_feedforwad)
            self.linear2 = torch.nn.Linear(dim_feedforwad, d_model)
            self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout = torch.nn.Dropout(dropout)
            self.dropout1 = torch.nn.Dropout(dropout)
            self.dropout2 = torch.nn.Dropout(dropout)

        def forward(self, x):
            # Attention, dropout, norm
            out = self.self_attn(x, x, x)[0]
            out = self.dropout(out)
            out = self.norm1(x + out)
            # Feedforward
            out_ff = self.linear1(out)
            out_ff = self.dropout1(out_ff)
            out_ff = torch.nn.functional.relu(out_ff)
            out_ff = self.linear2(out_ff)
            # Dropout, norm
            out_ff = self.dropout2(out_ff)
            out = self.norm2(out + out_ff)
            return out

    np.random.seed(0)
    torch.manual_seed(0)
    for _ in range(100):
        batch_size = np.random.randint(1, 5)
        seq_len = np.random.randint(1, 5)
        nhead = np.random.randint(1, 5)
        d_model = np.random.randint(1, 5) * nhead
        dim_feedforward = np.random.randint(1, 5)
        dropout = 0.0
        layer_norm_eps = 1e-5
        batch_first = True
        bias = True

        enc_man = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, layer_norm_eps=layer_norm_eps
        )
        enc_torch = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
        )

        # for n, p in enc_torch.named_parameters():
        #     print(n, p.shape)
        # print("\n")

        # The parameters are
        # self_attn.in_proj_weight torch.Size([6, 2])
        # self_attn.in_proj_bias torch.Size([6])
        # self_attn.out_proj.weight torch.Size([2, 2])
        # self_attn.out_proj.bias torch.Size([2])
        # linear1.weight torch.Size([5, 2])
        # linear1.bias torch.Size([5])
        # linear2.weight torch.Size([2, 5])
        # linear2.bias torch.Size([2])
        # norm1.weight torch.Size([2])
        # norm1.bias torch.Size([2])
        # norm2.weight torch.Size([2])
        # norm2.bias torch.Size([2])
        # equate all of them:
        enc_man.self_attn.in_proj_weight = enc_torch.self_attn.in_proj_weight
        enc_man.self_attn.in_proj_bias = enc_torch.self_attn.in_proj_bias
        enc_man.self_attn.out_proj.weight = enc_torch.self_attn.out_proj.weight
        enc_man.self_attn.out_proj.bias = enc_torch.self_attn.out_proj.bias
        # equate_torch_to_tocha_attention(enc_man.self_attn, enc_torch.self_attn, bias, nhead)

        enc_man.linear1.weight = enc_torch.linear1.weight
        enc_man.linear1.bias = enc_torch.linear1.bias
        enc_man.linear2.weight = enc_torch.linear2.weight
        enc_man.linear2.bias = enc_torch.linear2.bias
        enc_man.norm1.weight = enc_torch.norm1.weight
        enc_man.norm1.bias = enc_torch.norm1.bias
        enc_man.norm2.weight = enc_torch.norm2.weight
        enc_man.norm2.bias = enc_torch.norm2.bias

        # test
        x = torch.randn(batch_size, seq_len, d_model)
        enc_torch.eval()
        enc_man.eval()
        out1 = enc_torch(x)
        out2 = enc_man(x)
        passforward = torch.allclose(out1, out2, atol=1e-5)
        assert passforward
        
        if not passforward:
            print("failed") ; break
            continue
            print(f"""failed, parameters:
    {batch_size=}
    {seq_len=}
    {nhead=}
    {d_model=}
    {dim_feedforward=}
    {dropout=}
    {layer_norm_eps=}
    {batch_first=}
    {bias=}""")
        if passforward:
            continue
            print(f"""passed, parameters:
    {batch_size=}
    {seq_len=}
    {nhead=}
    {d_model=}
    {dim_feedforward=}
    {dropout=}
    {layer_norm_eps=}
    {batch_first=}
    {bias=}""")


def equate_torch_to_tocha_attention(torc, toch, bias, num_heads):
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


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforwad: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforwad)
        self.linear2 = Linear(dim_feedforwad, d_model)
        self.norm1 = LayerNorm((seq_len, d_model), eps=layer_norm_eps)
        self.norm2 = LayerNorm((seq_len, d_model), eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Attention, dropout, norm
        out = self.self_attn(x, x, x)
        out = self.dropout(out)
        # print(out.shape, x.shape, (x + out).shape, type(x), type(out))
        out = self.norm1(x + out)
        # Feedforward
        out_ff = self.linear1(out)
        out_ff = self.dropout1(out_ff)
        out_ff = F.relu(out_ff)
        out_ff = self.linear2(out_ff)
        # Dropout, norm
        out_ff = self.dropout2(out_ff)
        out = self.norm2(out + out_ff)
        return out

batch_size = 2
seq_len = 3
d_model = 2
nhead = 2
dim_feedforward = 5
dropout = 0.0
layer_norm_eps = 1e-5
batch_first = True
bias = True

enc_tocha = TransformerEncoderLayer(
    d_model, nhead, dim_feedforward, dropout, layer_norm_eps=layer_norm_eps
)
enc_torch = torch.nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    layer_norm_eps=layer_norm_eps,
    batch_first=batch_first,
)


# for n, p in enc_tocha.named_parameters():
#     print(n, type(p), p.shape)


equate_torch_to_tocha_attention(enc_torch.self_attn, enc_tocha.self_attn, bias, nhead)
# enc_man.self_attn.in_proj_weight = enc_torch.self_attn.in_proj_weight
# enc_man.self_attn.in_proj_bias = enc_torch.self_attn.in_proj_bias
# enc_man.self_attn.out_proj.weight = enc_torch.self_attn.out_proj.weight
# enc_man.self_attn.out_proj.bias = enc_torch.self_attn.out_proj.bias
# equate_torch_to_tocha_attention(enc_man.self_attn, enc_torch.self_attn, bias, nhead)

# COMPARE SHAPES:
print(f"{enc_tocha.linear1.weights.shape == enc_torch.linear1.weight.T.shape}")
print(f"{enc_tocha.linear1.bias.shape == enc_torch.linear1.bias.shape}")
print(f"{enc_tocha.linear2.weights.shape == enc_torch.linear2.weight.T.shape}")
print(f"{enc_tocha.linear2.bias.shape == enc_torch.linear2.bias.shape}")
print(f"{enc_tocha.norm1.weight.shape , enc_torch.norm1.weight.shape}")
print(f"{enc_tocha.norm1.bias.shape , enc_torch.norm1.bias.shape}")
print(f"{enc_tocha.norm2.weight.shape , enc_torch.norm2.weight.shape}")
print(f"{enc_tocha.norm2.bias.shape , enc_torch.norm2.bias.shape}")

enc_tocha.linear1.weights.data = enc_torch.linear1.weight.T.detach().numpy()
enc_tocha.linear1.bias.data = enc_torch.linear1.bias.detach().numpy()
enc_tocha.linear2.weights.data = enc_torch.linear2.weight.T.detach().numpy()
enc_tocha.linear2.bias.data = enc_torch.linear2.bias.detach().numpy()
enc_tocha.norm1.weight.data = enc_torch.norm1.weight.detach().numpy()
enc_tocha.norm1.bias.data = enc_torch.norm1.bias.detach().numpy()
enc_tocha.norm2.weight.data = enc_torch.norm2.weight.detach().numpy()
enc_tocha.norm2.bias.data = enc_torch.norm2.bias.detach().numpy()

# for n, p in enc_tocha.named_parameters():
#     print(n, type(p), p.shape)

xnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
x_tocha = tocha.tensor(xnp, requires_grad=True)
x_torch = torch.tensor(xnp, requires_grad=True)

out_tocha = enc_tocha(x_tocha)
out_torch = enc_torch(x_torch)

passforward = np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-5)
print(passforward)