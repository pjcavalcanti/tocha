from autograd.tensor import Tensor, Arrayable, ensure_array
from typing import Any, List, Tuple, Union
import numpy as np


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor]):
        super().__init__(
            data.data if isinstance(data, Tensor) else ensure_array(data),
            requires_grad=True,
        )


class Module:
    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def forward(self, *args) -> Tensor:
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def parameters(self) -> List[Parameter]:
        params = []
        for var in vars(self).items():
            if isinstance(var[1], Parameter):
                params.append(var[1])
            if isinstance(var[1], Module):
                params.extend(var[1].parameters())
        return params


class ParameterList(Module):
    def __init__(self, parameters: List[Parameter]):
        self.parameterlist = parameters

    def parameters(self) -> List[Parameter]:
        return self.parameterlist


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = Parameter(np.random.randn(in_features, out_features))
        self.bias = Parameter(np.random.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weights + self.bias
        return out


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.kernels = ParameterList(
            [
                Parameter(np.random.randn(kernel_size, in_channels))
                for _ in range(out_channels)
            ]
        )
        self.bias = Parameter(np.random.randn(out_channels))

    def forward(self, x: Tensor) -> Tensor:
        # Calculate the size of the output tensor
        out_size = x.shape[-1] - self.kernel_size + 1

        # Initialize the output tensor with zeros
        out = np.zeros((x.shape[0], self.out_channels, out_size))

        # Loop over each channel
        for i in range(self.out_channels):
            # Perform the 1D convolution
            for j in range(out_size):
                out[:, i, j] = (
                    x[:, :, j : j + self.kernel_size] * self.kernels.parameterlist[i]
                ).sum() + self.bias.data[i]

        return Tensor(out, requires_grad=True)
