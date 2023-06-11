import unittest
import numpy as np
import tocha
import torch

# class TestTransformerEncoder(unittest.TestCase):
#     # def test_against_torch(self):
#     #     n_head = 3
#     #     d_model = 5 * n_head
#     #     dim_feedforward = 7
#     #     dropout = 0.0
#     #     layer_norm_eps = 1e-5

#     #     num_layers = int(np.random.randint(1, 3))

#     #     batch_size = 2
#     #     seq_len = 3

#     #     enc_layer_torch = torch.nn.TransformerEncoderLayer(
#     #         d_model=d_model,
#     #         nhead=n_head,
#     #         dim_feedforward=dim_feedforward,
#     #         dropout=dropout,
#     #         layer_norm_eps=layer_norm_eps,
#     #         batch_first=True,
#     #     )
#     #     enc_torch = torch.nn.TransformerEncoder(
#     #         encoder_layer=enc_layer_torch, num_layers=num_layers
#     #     )
#     #     enc_layer_tocha = tocha.nn.TransformerEncoderLayer(
#     #         d_model=d_model,
#     #         dim_feedforwad=dim_feedforward,
#     #         nhead=n_head,
#     #         dropout=dropout,
#     #         layer_norm_eps=layer_norm_eps,
#     #     )
#     #     equate_tocha_to_torch_transformer_encoder_layer(enc_layer_tocha, enc_layer_torch)
#     #     enc_tocha = tocha.nn.TransformerEncoder(encoder_layer=enc_layer_tocha, num_layers=num_layers)


#     #     xnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
#     #     x_tocha = tocha.tensor(xnp, requires_grad=True)
#     #     x_torch = torch.tensor(xnp, requires_grad=True)

#     #     out_tocha = enc_tocha(x_tocha)
#     #     out_torch = enc_torch(x_torch)

#     #     passforward = np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-5)
#     #     assert passforward
        
#     #     grad = np.random.randn(*out_tocha.shape).astype(np.float32)
#     #     grad_tocha = tocha.tensor(grad)
#     #     grad_torch = torch.tensor(grad)
        
#     #     out_tocha.backward(grad_tocha)
#     #     out_torch.backward(grad_torch)
        
#     #     passbackward = np.allclose(x_tocha.grad.data, x_torch.grad.detach().numpy(), atol=1e-5)
#     #     assert passbackward
        
        
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
