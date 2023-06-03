from typing import Tuple
import torch
import tocha
from tocha import Tensor
from tocha.module import Module, Parameter
import numpy as np

b, c, h = 2, 3, 4
xnp = torch.randn(b, c, h)
x = torch.tensor(xnp, requires_grad=True)
normalized_shape = x.shape[1:]


norm = torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
for n, p in norm.named_parameters():
    print(n, p.shape)
out_torch = norm(x)

    
dims = tuple(len(x.shape) - i for i in range(len(normalized_shape), 0, -1))
mean = x.mean(dims, keepdim=True)
var = (x - mean).pow(2).mean(dims, keepdim=True)
out_man = (x - mean) / torch.sqrt(var + norm.eps)
out_man = out_man * norm.weight + norm.bias


class LayerNorm(Module):
    def __init__(self, normalized_shape: Tuple[int,...], eps: float=1e-5, elementwise_affine: bool=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.meandims = tuple(len(normalized_shape) - i for i in range(len(normalized_shape), 0, -1))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(np.ones(normalized_shape))
            self.bias = Parameter(np.zeros(normalized_shape))
            
        def forward(self, x: Tensor) -> Tensor:
            mean = x.mean(self.meandims, keepdims=True)
            var = ((x - mean) ** 2).mean(self.meandims, keepdims=True)
            out = (x - mean) / (var + self.eps).sqrt()

print(torch.allclose(out_man, out_torch, atol=1e-6))

