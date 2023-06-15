import torch
import numpy as np

import tocha
import tocha.functional as F
from tocha import Tensor
from tocha.module import Module
from tocha.nn import LayerNorm, Dropout, Linear, MultiheadAttention


def equate_tocha_to_torch_linear(toch, torc):
    toch.weights.data = torc.weight.T.detach().numpy().copy()
    toch.bias.data = torc.bias.detach().numpy().copy()


def equate_tocha_to_torch_layer_norm(toch, torc):
    toch.weight.data = torc.weight.detach().numpy().copy()
    toch.bias.data = torc.bias.detach().numpy().copy()


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
        head.q_proj_weight.data = wq.detach().numpy().copy()
        head.k_proj_weight.data = wk.detach().numpy().copy()
        head.v_proj_weight.data = wv.detach().numpy().copy()
    if bias:
        for h, (bq, bk, bv) in enumerate(zip(head_bq, head_bk, head_bv)):
            head = getattr(toch, f"head_{h}")
            head.q_proj_bias.data = bq.detach().numpy().copy()
            head.k_proj_bias.data = bk.detach().numpy().copy()
            head.v_proj_bias.data = bv.detach().numpy().copy()
    toch.out_proj_weight.data = out_weight.t().detach().numpy().copy()
    if bias:
        toch.out_proj_bias.data = out_bias.detach().numpy().copy()


def equate_tocha_to_torch_transformer_encoder_layer(toch, torc):
    equate_tocha_to_torch_attention(toch.self_attn, torc.self_attn, True, toch.nhead)

    toch.linear1.weights.data = torc.linear1.weight.T.detach().numpy().copy()
    toch.linear1.bias.data = torc.linear1.bias.detach().numpy().copy()
    toch.linear2.weights.data = torc.linear2.weight.T.detach().numpy().copy()
    toch.linear2.bias.data = torc.linear2.bias.detach().numpy().copy()
    toch.norm1.weight.data = torc.norm1.weight.detach().numpy().copy()
    toch.norm1.bias.data = torc.norm1.bias.detach().numpy().copy()
    toch.norm2.weight.data = torc.norm2.weight.detach().numpy().copy()
    toch.norm2.bias.data = torc.norm2.bias.detach().numpy().copy()

def equate_tocha_to_torch_transformer_decoder_layer(toch, torc, nhead):
    equate_tocha_to_torch_attention(
        toch.self_attn, torc.self_attn, bias=True, num_heads=nhead
    )
    equate_tocha_to_torch_attention(
        toch.multihead_attn, torc.multihead_attn, bias=True, num_heads=nhead
    )
    equate_tocha_to_torch_linear(toch.linear1, torc.linear1)
    equate_tocha_to_torch_linear(toch.linear2, torc.linear2)
    equate_tocha_to_torch_layer_norm(toch.norm1, torc.norm1)
    equate_tocha_to_torch_layer_norm(toch.norm2, torc.norm2)
    equate_tocha_to_torch_layer_norm(toch.norm3, torc.norm3)

def reconstruct_torch_decoder():
    nhead = 2
    d_model = 3 * nhead
    dim_feedforward = 5
    dropout = 0.0
    layer_norm_eps = 1e-5

    torch_dec = torch.nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        batch_first=True,
    )

    # for n, p in torch_dec.named_parameters():
    #     print(n, p.shape)

    class TransformerDecoderLayer(torch.nn.Module):
        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            layer_norm_eps: float,
        ) -> None:
            super().__init__()
            assert d_model % nhead == 0, "d_model must be divisible by nhead"
            self.d_model = d_model
            self.nhead = nhead
            self.dim_feedforward = dim_feedforward
            self.layer_norm_eps = layer_norm_eps

            self.head_dim = d_model // nhead

            self.self_attn = torch.nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.multihead_attn = torch.nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

            self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

            self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm3 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

            self.activation = torch.nn.ReLU()

        def forward(self, tgt, memory):
            # First (masked) multihead attention on the input
            out1 = self.self_attn(tgt, tgt, tgt)[0]
            out1 = self.norm1(out1 + tgt)

            # Second multihead attention on the encoder output and the output of the first attention
            out2 = self.multihead_attn(out1, memory, memory)[
                0
            ]  # Different order from paper
            out2 = self.norm2(out2 + out1)

            # Feedforward
            out3 = self.activation(self.linear1(out2))
            out3 = self.linear2(out3)
            out4 = self.norm3(out3 + out2)

            return out4

    my_dec = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        layer_norm_eps=layer_norm_eps,
    )

    my_dec.self_attn.in_proj_weight = torch_dec.self_attn.in_proj_weight
    my_dec.self_attn.in_proj_bias = torch_dec.self_attn.in_proj_bias
    my_dec.self_attn.out_proj.weight = torch_dec.self_attn.out_proj.weight
    my_dec.self_attn.out_proj.bias = torch_dec.self_attn.out_proj.bias

    my_dec.multihead_attn.in_proj_weight = torch_dec.multihead_attn.in_proj_weight
    my_dec.multihead_attn.in_proj_bias = torch_dec.multihead_attn.in_proj_bias
    my_dec.multihead_attn.out_proj.weight = torch_dec.multihead_attn.out_proj.weight
    my_dec.multihead_attn.out_proj.bias = torch_dec.multihead_attn.out_proj.bias

    my_dec.linear1.weight = torch_dec.linear1.weight
    my_dec.linear1.bias = torch_dec.linear1.bias

    my_dec.linear2.weight = torch_dec.linear2.weight
    my_dec.linear2.bias = torch_dec.linear2.bias

    my_dec.norm1.weight = torch_dec.norm1.weight
    my_dec.norm1.bias = torch_dec.norm1.bias

    my_dec.norm2.weight = torch_dec.norm2.weight
    my_dec.norm2.bias = torch_dec.norm2.bias

    batch_size = 7
    seq_len = 11

    x = torch.rand(batch_size, seq_len, d_model)

    torch_dec.eval()
    my_dec.eval()

    out_torch = torch_dec(x, x)
    out_my = my_dec(x, x)

    equal = torch.allclose(out_torch, out_my)
    print(equal)


def decoder_layer_with_tocha():
    class TransformerDecoderLayer(Module):
        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            layer_norm_eps: float = 1e-5,
            dropout: float = 0.1,
        ):
            super().__init__()
            assert d_model % nhead == 0, "d_model must be divisible by nhead"
            assert dropout >= 0 and dropout <= 1, "dropout must be between 0 and 1"
            self.d_model = d_model
            self.nhead = nhead
            self.dim_feedforward = dim_feedforward
            self.head_dim = d_model // nhead

            self.self_attn = MultiheadAttention(d_model, nhead)
            self.multihead_attn = MultiheadAttention(d_model, nhead)

            self.linear1 = Linear(d_model, dim_feedforward)
            self.linear2 = Linear(dim_feedforward, d_model)

            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
            self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

            self.dropout = Dropout(dropout)

        def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
            # First attention on input
            out1 = self.self_attn(tgt, tgt, tgt)
            out1 = self.dropout(out1)
            out1 = self.norm1(out1 + tgt)

            # Second attention on encoder output and output of first attention
            out2 = self.multihead_attn(out1, memory, memory)
            out2 = self.dropout(out2)
            out2 = self.norm2(out2 + out1)

            # Feedforward
            out3 = self.linear1(out2)
            out3 = self.dropout(out3)
            out3 = F.relu(out3)
            out3 = self.linear2(out3)
            out4 = self.dropout(out3)
            out4 = self.norm3(out4 + out2)

            return out4

    np.random.seed(0)
    torch.manual_seed(0)
    for _ in range(100):
        nhead = np.random.randint(1, 4)
        d_model = np.random.randint(1, 4) * nhead
        dim_feedforward = np.random.randint(1, 4)
        dropout = 0.0
        layer_norm_eps = np.random.randint(1, 4) * 10 ** -np.random.randint(1, 6)
        
        batch_size = np.random.randint(1, 4)
        seq_len = np.random.randint(1, 4)

        dec_torch = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            batch_first=True,
        )
        dec_tocha = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )

        equate_tocha_to_torch_transformer_decoder_layer(dec_tocha, dec_torch, nhead)

        tgtnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        memorynp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        tgt_tocha = tocha.tensor(tgtnp, requires_grad=True)
        tgt_torch = torch.tensor(tgtnp, requires_grad=True)
        memory_tocha = tocha.tensor(memorynp, requires_grad=True)
        memory_torch = torch.tensor(memorynp, requires_grad=True)

        dec_tocha.eval()
        dec_torch.eval()

        out_tocha = dec_tocha(tgt_tocha, memory_tocha)
        out_torch = dec_torch(tgt_torch, memory_torch)

        passforward = np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-5)
        assert passforward, "forward pass failed"

        gradnp = np.random.randn(*out_torch.shape).astype(np.float32)
        grad_tocha = tocha.tensor(gradnp)
        grad_torch = torch.tensor(gradnp)
        
        out_tocha.backward(grad_tocha)
        out_torch.backward(grad_torch)
        
        passgradtgt = np.allclose(tgt_tocha.grad.data, tgt_torch.grad.detach().numpy(), atol=1e-5)
        passgradmem = np.allclose(memory_tocha.grad.data, memory_torch.grad.detach().numpy(), atol=1e-5)
        
        assert passgradtgt, "tgt grad failed"
        assert passgradmem, "mem grad failed"
        print(f"pass {_}")


decoder_layer_with_tocha()
