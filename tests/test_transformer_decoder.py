import unittest
import numpy as np
import tocha
import torch

# THIS IS JUST TOO MUCH SLOW

# class TestTransformerEncoder(unittest.TestCase):
#     def test_against_torch(self):
#         np.random.seed(0)
#         torch.manual_seed(0)
#         for _ in range(1):
#             nhead = 2 # np.random.randint(1, 10)
#             d_model = 3 * nhead #np.random.randint(1, 10) * nhead
#             num_layers = 2 #np.random.randint(1, 10)
#             dim_feedforward = 3 # np.random.randint(1, 10)
#             dropout = 0.0
#             layer_norm_eps = 1e-5 #np.random.random() *10 **-np.random.randint(1, 10)

#             batch_size = 3 #np.random.randint(1, 10)
#             seq_len = 4 #np.random.randint(1, 10)

#             tgtnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
#             memorynp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

#             tgt_tocha = tocha.tensor(tgtnp, requires_grad=True)
#             tgt_torch = torch.tensor(tgtnp, requires_grad=True)
#             memory_tocha = tocha.tensor(memorynp, requires_grad=True)
#             memory_torch = torch.tensor(memorynp, requires_grad=True)

#             dec_layer_torch = torch.nn.TransformerDecoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=dim_feedforward,
#                 dropout=dropout,
#                 layer_norm_eps=layer_norm_eps,
#                 batch_first=True,
#             )
#             dec_torch = torch.nn.TransformerDecoder(dec_layer_torch, num_layers=num_layers)

#             dec_layer_tocha = tocha.nn.TransformerDecoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=dim_feedforward,
#                 dropout=dropout,
#                 layer_norm_eps=layer_norm_eps,
#             )
#             dec_tocha = tocha.nn.TransformerDecoder(dec_layer_tocha, num_layers=num_layers)

#             for l, layer in enumerate(dec_tocha.layers):
#                 equate_tocha_to_torch_transformer_decoder_layer(layer, dec_torch.layers[l], nhead)
                
#             out_torch = dec_torch(tgt_torch, memory_torch)
#             out_tocha = dec_tocha(tgt_tocha, memory_tocha)

#             passforward = np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-5)
#             assert passforward
            
#             gradnp = np.random.randn(*out_torch.shape).astype(np.float32)
#             grad_tocha = tocha.tensor(gradnp)
#             grad_torch = torch.tensor(gradnp)
            
#             out_tocha.backward(grad_tocha)
#             out_torch.backward(grad_torch)
            
#             passbackward_tgt = np.allclose(tgt_tocha.grad.data, tgt_torch.grad.detach().numpy(), atol=1e-5)
#             passbackward_memory = np.allclose(memory_tocha.grad.data, memory_torch.grad.detach().numpy(), atol=1e-5)
            
#             assert passbackward_tgt
#             assert passbackward_memory
        
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
