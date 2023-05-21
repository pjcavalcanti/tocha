import tocha
import tocha.functional as F
from tocha.module import Module, Parameter
import numpy as np

from typing import Tuple
from autograd.tensor import Tensor, Arrayable, ensure_array


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = Parameter(np.random.randn(in_features, out_features))
        self.bias = Parameter(np.random.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weights + self.bias
        return out


class Conv2d(Module):
    # TODO: Add padding and stride
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Tuple[int, ...],
        bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.bias = None

        self.weight = Parameter(
            np.random.randn(out_features, in_features, kernel_size[0] * kernel_size[1])
        )
        if bias:
            self.bias = Parameter(np.random.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        assert (
            len(x.shape) == 4
        ), "Input tensor must be (batch_size, channels, height, width)"
        # Separate submatrices with im2col
        out = F.im2col2d(x, self.kernel_size)

        # Apply convolution
        #  out    = # (B, Cin, k1*k2, (H - k1 + 1)*(W - k2 + 1))
        #  weight = # (Cout, Cin, k1*k2)
        out = tocha.tensordot(out, self.weight, axes=((-3, -2), (1, 2)))

        # Add bias
        # out =  # (B, (H - k1 + 1)*(W - k2 + 1),  Cout)
        # bias = # (Cout,)
        if self.bias is not None:
            out = out + self.bias

        # Transpose to get the right index order
        axes = tuple(range(len(out.shape)))
        axes = axes[:-2] + (axes[-1], axes[-2])
        out = tocha.transpose(out, axes=axes)
        # out = (B, Cout, (H - k1 + 1)*(W - k2 + 1))

        # Reshape to get the right output shape
        batch = x.shape[0]
        x_out = x.shape[-2] - self.kernel_size[0] + 1
        y_out = x.shape[-1] - self.kernel_size[1] + 1
        out = out.reshape((batch, self.out_features, x_out, y_out))
        return out
