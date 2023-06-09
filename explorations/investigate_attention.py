from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

import tocha
from tocha import Tensor
import tocha.nn as nn
import tocha.functional as F

torch.manual_seed(0)


def how_torch_does_it():
    num_heads = 3
    embed_dim = 3 * num_heads
    batch_size = 5
    seq_len = 7
    dropout = 0.0
    x_torch = torch.randn(batch_size, seq_len, embed_dim)

    attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
    attention.eval()

    qkv_weight = attention.in_proj_weight
    qkv_bias = attention.in_proj_bias
    out_weight = attention.out_proj.weight
    out_bias = attention.out_proj.bias

    # q_weight = qkv_weight[:embed_dim, :]
    # k_weight = qkv_weight[embed_dim:2*embed_dim, :]
    # v_weight = qkv_weight[2*embed_dim:, :]
    q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)

    # q_bias = qkv_bias[:embed_dim]
    # k_bias = qkv_bias[embed_dim:2*embed_dim]
    # v_bias = qkv_bias[2*embed_dim:]
    q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

    query_proj = torch.matmul(x_torch, q_weight.t()) + q_bias
    key_proj = torch.matmul(x_torch, k_weight.t()) + k_bias
    value_proj = torch.matmul(x_torch, v_weight.t()) + v_bias

    head_dim = embed_dim // num_heads
    print(f"{batch_size}, {seq_len}, {num_heads}, {head_dim}")
    print(f"{query_proj.reshape(batch_size, -1, num_heads, head_dim).shape}")
    query_proj = query_proj.reshape(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    key_proj = key_proj.reshape(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    value_proj = value_proj.reshape(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    scaling_factor = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
    attention_scores = (
        torch.matmul(query_proj, key_proj.transpose(-2, -1)) / scaling_factor
    )
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_probs, value_proj)

    output = output.transpose(1, 2).contiguous().view(batch_size, -1, embed_dim)
    output = torch.matmul(output, out_weight.t()) + out_bias

    out_torch, weights_torch = attention(x_torch, x_torch, x_torch)

    print(out_torch.shape, output.shape)
    print(torch.allclose(output, out_torch, atol=1e-4))


def more_explicit_separated_heads():
    torch.manual_seed(0)
    for _ in range(100):
        num_heads = np.random.randint(1, 5)
        embed_dim = np.random.randint(1, 5) * num_heads
        batch_size = np.random.randint(1, 5)
        seq_len = np.random.randint(1, 5)
        dropout = 0.0
        bias = bool(np.random.choice([True, False]))
        x_torch = torch.randn(batch_size, seq_len, embed_dim)

        attention = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout, batch_first=True, bias=bias
        )
        attention.eval()

        qkv_weight = attention.in_proj_weight
        out_weight = attention.out_proj.weight
        if bias:
            qkv_bias = attention.in_proj_bias
            out_bias = attention.out_proj.bias

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

        # project x
        if bias:
            queries_proj = [x_torch @ wq + bq for wq, bq in zip(head_wq, head_bq)]
            keys_proj = [x_torch @ wk + bk for wk, bk in zip(head_wk, head_bk)]
            values_proj = [x_torch @ wv + bv for wv, bv in zip(head_wv, head_bv)]
        else:
            queries_proj = [x_torch @ wq for wq in head_wq]
            keys_proj = [x_torch @ wk for wk in head_wk]
            values_proj = [x_torch @ wv for wv in head_wv]

        # compute attention
        head_dim = embed_dim // num_heads
        scaling_factor = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        head_attentions = [
            torch.nn.functional.softmax(
                (q @ k.transpose(-2, -1)) / scaling_factor, dim=-1
            )
            @ v
            for q, k, v in zip(queries_proj, keys_proj, values_proj)
        ]
        # print(head_attentions[0].shape)
        head_attentions = torch.concat(
            head_attentions, dim=-1
        )  # -1 is the final "head_dim" dimension
        if bias:
            output = head_attentions @ out_weight.t() + out_bias
        else:
            output = head_attentions @ out_weight.t()

        # compare to torch
        out_torch, weights_torch = attention(x_torch, x_torch, x_torch)
        # print(out_torch.shape, output.shape)
        print(torch.allclose(output, out_torch, atol=1e-6))


class ScaledDotProductAttentionHead(nn.Module):
    # Note the projection is absorbed here
    def __init__(self, embed_dim: int, head_dim: int, bias: bool) -> None:
        self.head_dim = head_dim
        self.num_heads = embed_dim // head_dim
        self.bias = bias
        self.scale = np.sqrt(head_dim)

        self.q_proj_weight = nn.Parameter(np.random.randn(embed_dim, head_dim))
        self.k_proj_weight = nn.Parameter(np.random.randn(embed_dim, head_dim))
        self.v_proj_weight = nn.Parameter(np.random.randn(embed_dim, head_dim))

        if self.bias:
            self.q_proj_bias = nn.Parameter(np.random.randn(head_dim))
            self.k_proj_bias = nn.Parameter(np.random.randn(head_dim))
            self.v_proj_bias = nn.Parameter(np.random.randn(head_dim))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        # use attention formula from https://arxiv.org/pdf/1706.03762.pdf
        Q = q @ self.q_proj_weight
        K = k @ self.k_proj_weight
        V = v @ self.v_proj_weight
        if self.bias:
            Q += self.q_proj_bias
            K += self.k_proj_bias
            V += self.v_proj_bias

        att = Q @ K.transpose((0, 2, 1)) / self.scale
        if att_mask is not None:
            att = att_mask * att
        att = F.softmax(Q @ K.transpose((0, 2, 1)) / self.scale, dim=-1) @ V
        return att


class MultiHeadAttention(nn.Module):
    # Assumes that the input has batch_first = True
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert dropout >= 0.0 and dropout <= 1.0, "dropout must be between 0.0 and 1.0"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

        self.head_dim = embed_dim // num_heads

        for i in range(num_heads):
            new_head = ScaledDotProductAttentionHead(embed_dim, self.head_dim, bias)
            self.register_module(f"head_{i}", new_head)
        self.out_proj_weight = nn.Parameter(np.random.randn(embed_dim, embed_dim))
        if self.bias:
            self.out_proj_bias = nn.Parameter(np.random.randn(embed_dim))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        # apply all heads
        head_outputs = [getattr(self, f"head_{i}")(q, k, v, attn_mask) for i in range(self.num_heads)]
        # concatenate head outputs, then project
        output = tocha.concatenate(head_outputs, axis=-1) @ self.out_proj_weight
        if self.bias:
            output += self.out_proj_bias
        if self.dropout is not None:
            output = self.dropout(output)
        return output


def equate_torch_to_tocha_attention(tor, toc, bias, num_heads):
    qkv_weight = tor.in_proj_weight
    out_weight = tor.out_proj.weight
    if bias:
        qkv_bias = tor.in_proj_bias
        out_bias = tor.out_proj.bias

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
        head = getattr(toc, f"head_{h}")
        head.q_proj_weight.data = wq.detach().numpy()
        head.k_proj_weight.data = wk.detach().numpy()
        head.v_proj_weight.data = wv.detach().numpy()
    if bias:
        for h, (bq, bk, bv) in enumerate(zip(head_bq, head_bk, head_bv)):
            head = getattr(toc, f"head_{h}")
            head.q_proj_bias.data = bq.detach().numpy()
            head.k_proj_bias.data = bk.detach().numpy()
            head.v_proj_bias.data = bv.detach().numpy()
    toc.out_proj_weight.data = out_weight.t().detach().numpy()
    if bias:
        toc.out_proj_bias.data = out_bias.detach().numpy()
    

for _ in range(100):
    np.random.seed(0)
    num_heads = np.random.randint(1, 5)
    embed_dim = np.random.randint(1, 5) * num_heads
    batch_size = np.random.randint(1, 5)
    seq_len = np.random.randint(1, 5)
    dropout = 0.0
    bias = bool(np.random.choice([True, False]))

    xnp = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    qnp = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    knp = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    vnp = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    q_tocha = tocha.tensor(qnp, requires_grad=True)
    k_tocha = tocha.tensor(knp, requires_grad=True)
    v_tocha = tocha.tensor(vnp, requires_grad=True)
    q_torch = torch.tensor(qnp, requires_grad=True)
    k_torch = torch.tensor(knp, requires_grad=True)
    v_torch = torch.tensor(vnp, requires_grad=True)

    # masknp = np.tril(-np.inf * np.ones([num_heads,num_heads]), k=-1).T # upper triangular
    masknp = np.random.randn(num_heads, num_heads).astype(np.float32).T
    mask_tocha = tocha.tensor(masknp, requires_grad=False)
    mask_torch = torch.tensor(masknp, requires_grad=False)

    
    attention_tocha = MultiHeadAttention(embed_dim, num_heads, bias=bias, dropout=dropout)
    attention_torch = torch.nn.MultiheadAttention(
        embed_dim, num_heads, dropout, batch_first=True, bias=bias
    )
    attention_torch.eval()

    equate_torch_to_tocha_attention(attention_torch, attention_tocha, bias, num_heads)
    # qkv_weight = attention_torch.in_proj_weight
    # out_weight = attention_torch.out_proj.weight
    # if bias:
    #     qkv_bias = attention_torch.in_proj_bias
    #     out_bias = attention_torch.out_proj.bias

    # # separate q,k,v
    # q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
    # if bias:
    #     q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

    # # separate head weights/biases
    # head_wq = q_weight.t().chunk(dim=1, chunks=num_heads)
    # head_wk = k_weight.t().chunk(dim=1, chunks=num_heads)
    # head_wv = v_weight.t().chunk(dim=1, chunks=num_heads)
    # if bias:
    #     head_bq = q_bias.chunk(dim=0, chunks=num_heads)
    #     head_bk = k_bias.chunk(dim=0, chunks=num_heads)
    #     head_bv = v_bias.chunk(dim=0, chunks=num_heads)

    # for h, (wq, wk, wv) in enumerate(zip(head_wq, head_wk, head_wv)):
    #     head = getattr(attention_tocha, f"head_{h}")
    #     head.q_proj_weight.data = wq.detach().numpy()
    #     head.k_proj_weight.data = wk.detach().numpy()
    #     head.v_proj_weight.data = wv.detach().numpy()
    # if bias:
    #     for h, (bq, bk, bv) in enumerate(zip(head_bq, head_bk, head_bv)):
    #         head = getattr(attention_tocha, f"head_{h}")
    #         head.q_proj_bias.data = bq.detach().numpy()
    #         head.k_proj_bias.data = bk.detach().numpy()
    #         head.v_proj_bias.data = bv.detach().numpy()
    # attention_tocha.out_proj_weight.data = out_weight.t().detach().numpy()
    # if bias:
    #     attention_tocha.out_proj_bias.data = out_bias.detach().numpy()

    out_tocha = attention_tocha(q_tocha, k_tocha, v_tocha, attn_mask=mask_tocha)
    out_torch = attention_torch(q_torch, k_torch, v_torch, attn_mask=mask_torch)[0]

    assert np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-6), "forward pass failed"
    
    gradnp = np.random.randn(*out_tocha.shape).astype(np.float32)
    grad_tocha = tocha.tensor(gradnp)
    grad_torch = torch.tensor(gradnp)

    out_tocha.backward(grad_tocha)
    out_torch.backward(grad_torch)
    
    assert np.allclose(q_tocha.grad.data, q_torch.grad.detach().numpy(), atol=1e-6), "backward pass failed at q"
    assert np.allclose(k_tocha.grad.data, k_torch.grad.detach().numpy(), atol=1e-6), "backward pass failed at k"
    assert np.allclose(v_tocha.grad.data, v_torch.grad.detach().numpy(), atol=1e-6), "backward pass failed at v"
