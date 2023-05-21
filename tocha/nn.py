from autograd.tensor import Tensor, Arrayable, ensure_array, tensordot
import tocha.functional as F
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


class Conv2d(Module):
    def __init__(self, in_features, out_features, kernel_size, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size

        self.weight = Parameter(
            np.random.randn(out_features, in_features, kernel_size[0] * kernel_size[1])
        )
        self.bias = Parameter(np.random.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        print("Begin forward")
        print(f"Input: {x.shape}")
        print(f"Parameters: {self.weight.shape}, {self.bias.shape}")
        out = F.im2col2d(x, self.kernel_size)
        # print(out.shape)
        out = tensordot(
            out, self.weight, axes=((-3, -2), (1, 2))
        )  # (B, Cin, W, H) (Cout, Cin, K1*K2)
        # print(out.shape)
        b_out = x.shape[0]
        c_out = self.weight.shape[0]
        x_out = x.shape[-2] - self.kernel_size[0] + 1
        y_out = x.shape[-1] - self.kernel_size[1] + 1
        out = out.reshape((b_out, c_out, x_out, y_out)) + self.bias  # (2,2,2,6)
        print(f"Output: {out.shape}\n")
        print("End forward")
        return out
