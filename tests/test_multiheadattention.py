import unittest
import numpy as np
import torch

import tocha
from tocha.nn import MultiheadAttention

class TestMultiheadAttention(unittest.TestCase):
    def test_multiheadattention_against_torch(self):
        for _ in range(100):
            np.random.seed(0)
            num_heads = np.random.randint(1, 5)
            embed_dim = np.random.randint(1, 5) * num_heads
            batch_size = np.random.randint(1, 5)
            seq_len = np.random.randint(1, 5)
            dropout = 0.0
            bias = bool(np.random.choice([True, False]))

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

            
            attention_tocha = MultiheadAttention(embed_dim, num_heads, bias=bias, dropout=dropout)
            attention_torch = torch.nn.MultiheadAttention(
                embed_dim, num_heads, dropout, batch_first=True, bias=bias
            )
            attention_torch.eval()

            qkv_weight = attention_torch.in_proj_weight
            out_weight = attention_torch.out_proj.weight
            if bias:
                qkv_bias = attention_torch.in_proj_bias
                out_bias = attention_torch.out_proj.bias

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
                head = getattr(attention_tocha, f"head_{h}")
                head.q_proj_weight.data = wq.detach().numpy()
                head.k_proj_weight.data = wk.detach().numpy()
                head.v_proj_weight.data = wv.detach().numpy()
            if bias:
                for h, (bq, bk, bv) in enumerate(zip(head_bq, head_bk, head_bv)):
                    head = getattr(attention_tocha, f"head_{h}")
                    head.q_proj_bias.data = bq.detach().numpy()
                    head.k_proj_bias.data = bk.detach().numpy()
                    head.v_proj_bias.data = bv.detach().numpy()
            attention_tocha.out_proj_weight.data = out_weight.t().detach().numpy()
            if bias:
                attention_tocha.out_proj_bias.data = out_bias.detach().numpy()

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
