from autograd.tensor import Tensor, Arrayable, ensure_array
from typing import Any, List, Union
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

    def parameters(self) -> List[Parameter]:
        params = []
        for var in vars(self).items():
            if isinstance(var[1], Parameter):
                params.append(var[1])
            if isinstance(var[1], Module):
                params.extend(var[1].parameters())
        return params


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = Parameter(np.random.randn(in_features, out_features))
        self.bias = Parameter(np.random.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weights + self.bias
        return out
