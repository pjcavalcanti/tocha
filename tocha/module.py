from typing import Any, List, Union
from autograd.tensor import Tensor, Arrayable, ensure_array


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor]):
        super().__init__(
            data.data if isinstance(data, Tensor) else ensure_array(data),
            requires_grad=True,
        )


class Module:
    def __init__(self):
        self.training = True
        
    def eval(self) -> None:
        self.training = False
        for child in self.children():
            child.eval()
    def train(self) -> None:
        self.training = True
        for child in self.children():
            child.train()
    
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

    def children(self) -> List["Module"]:
        child = []
        for var in vars(self).items():
            if isinstance(var[1], Module):
                child.append(var[1])
        return child


class ParameterList(Module):
    def __init__(self, parameters: List[Parameter]):
        self.parameterlist = parameters

    def parameters(self) -> List[Parameter]:
        return self.parameterlist
