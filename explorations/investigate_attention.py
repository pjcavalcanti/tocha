import torch
import torch.nn.functional as F


num_heads = 1
embed_dim = 2 * num_heads
dropout = 0.0
bias = True
batch_first = True
att = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, batch_first=batch_first)
att.eval()
for n, p in att.named_parameters():
    print(n, p.shape)
print()

batch_size = 1
seq_len = 3
x_torch = torch.randn(batch_size, seq_len, embed_dim)

out_torch = att(x_torch, x_torch, x_torch)

weight_q, weight_k, weight_v = torch.chunk(att.in_proj_weight, 3, dim=0)
if bias:
    bias_q, bias_k, bias_v = torch.chunk(att.in_proj_bias, 3, dim=0)
weight_out = att.out_proj.weight
bias_out = att.out_proj.bias

Q = (x_torch @ weight_q + bias_q)
Kt = (x_torch @ weight_k + bias_k).transpose(-2,-1)
V = x_torch @ weight_v + bias_v
att = F.softmax(Q @ Kt / (embed_dim ** 0.5), dim=-1) @ V
att = att @ weight_out.transpose(-2,-1) + bias_out


print(att.shape)
print(type(out_torch), len(out_torch))
print(out_torch[0].shape)
print(out_torch[1].shape)

print(torch.allclose(att, out_torch[0], atol=1e-4))
distance = torch.norm(att - out_torch[0])
print(f"distance = {distance}")