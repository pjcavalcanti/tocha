from typing import List, Tuple, Union
import torch
import tocha
from tocha import Tensor
from tocha.module import Module, Parameter
import numpy as np


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[List[int], int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (
            list([int(normalized_shape)])
            if isinstance(normalized_shape, int)
            else normalized_shape
        )
        print(type(self.normalized_shape))
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape))
            self.bias = Parameter(np.zeros(self.normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(len(x.shape) - i for i in range(len(self.normalized_shape), 0, -1))
        mean = x.mean(dims, keepdims=True)
        var = ((x - mean) ** 2).mean(dims, keepdims=True)
        out = (x - mean) / (var + self.eps).sqrt()
        if self.elementwise_affine:
            out = self.weight * out + self.bias
        return out


np.random.seed(0)
for _ in range(100):
    # prepare data
    ndims = np.random.randint(2, 10)
    shape = np.random.randint(1, 5, size=ndims)
    eps = np.random.rand() * 10 ** (np.random.randint(-10, -1))
    elementwise_affine = bool(np.random.choice([True, False]))

    xnp = np.random.randn(*shape).astype(np.float64)  # type: ignore
    i0 = np.random.randint(0, len(shape))
    normalized_shape = list(xnp.shape[i0:])
    if np.random.random() < 0.5:
        normalized_shape = int(shape[-1])
    x_tocha = tocha.tensor(xnp, requires_grad=True)
    x_torch = torch.tensor(xnp, requires_grad=True)

    # initialize and equate layers
    norm_torch = torch.nn.LayerNorm(
        normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
        dtype=torch.float64,
    )
    norm_tocha = LayerNorm(
        normalized_shape, eps=eps, elementwise_affine=elementwise_affine
    )
    if elementwise_affine:
        norm_tocha.weight.data = norm_torch.weight.detach().numpy()
        norm_tocha.bias.data = norm_torch.bias.detach().numpy()
    out_tocha = norm_tocha(x_tocha)
    out_torch = norm_torch(x_torch)

    # check forward and backward pass
    passforward = np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-10)
    assert passforward, "forward pass failed"

    gradnp = np.random.randn(*out_tocha.shape).astype(np.float64)  # type: ignore
    grad_tocha = tocha.tensor(gradnp)
    grad_torch = torch.tensor(gradnp)

    out_tocha.backward(grad_tocha)
    out_torch.backward(grad_torch)

    passbackward = np.allclose(x_tocha.grad.data, x_torch.grad.detach().numpy(), atol=1e-10)  # type: ignore
    assert passbackward, "backward pass failed"
