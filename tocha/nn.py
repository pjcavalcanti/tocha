import tocha
import tocha.functional as F
from tocha.module import Module, Parameter
import numpy as np

from typing import List, Tuple
from autograd.tensor import Tensor, Arrayable, ensure_array

## Non-linearities

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)
    
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)

class Tanh(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)
    
class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis
    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, dim = self.axis)

## Basic layers

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = Parameter(np.random.randn(in_features, out_features) / np.sqrt(in_features))
        self.bias = Parameter(np.random.randn(out_features) / np.sqrt(in_features))

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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.bias = None

        self.weight = Parameter(
            np.random.randn(out_features, in_features, kernel_size[0] * kernel_size[1]) / np.sqrt(in_features)
        )
        if bias:
            self.bias = Parameter(np.random.randn(out_features) / np.sqrt(in_features))

    def forward(self, x: Tensor) -> Tensor:
        assert (
            len(x.shape) == 4
        ), "Input tensor must be (batch_size, channels, height, width)"
        # Separate submatrices with im2col
        out = F.im2col(x, self.kernel_size)
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

## Containers

class Sequential(Module):
    def __init__(self, layers: List[Module]) -> None:
        super().__init__()
        self.layers = layers
        
    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

## Regularization layers

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p # probability of dropping a number
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask_np = np.random.binomial(1,1 - self.p,x.shape) 
            mask = tocha.tensor(mask_np, requires_grad=False)
            return mask * x / (1 - self.p)
        else:
            return x
        
class BatchNorm1d(Module):
    def __init__(self, num_features:int, eps: float = 1e-5, momentum: float=0.1) -> None:
        # here i use dim rather than num_features, which deviates from pytorch
        # this is because i want to avoid reshape in forward
        # so i can declare gamma and beta with the right shape in init
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))
        
        self.running_mean = Tensor(np.zeros(num_features), requires_grad=False)
        self.running_std = Tensor(np.ones(num_features), requires_grad=False)
        
    def forward(self, x: Tensor) -> Tensor:
        axis = 0
        if len(x.shape) == 3:
            axis = (0, 2)
        if self.training:
            # Normalize, scale and shift
            mean = x.mean(axis=axis, keepdims=True)
            std = (x - mean) ** 2
            std = std.mean(axis=axis, keepdims=True)
            std = std + self.eps
            std = std ** 0.5
            out = (x - mean) / std
            out = self.gamma * out + self.beta
            # Update running mean and variance
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_std.data = (1 - self.momentum) * self.running_std.data + self.momentum * std.data
        else:
            # Normalize, scale and shift
            out = (x - self.running_mean) / (self.running_std + self.eps)
            out = self.gamma * out + self.beta
        return out