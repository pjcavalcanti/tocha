from typing import Any, List, Union, Dict, Optional
from autograd.tensor import Tensor, Arrayable, ensure_array


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor], name: Optional[str] = None):
        super().__init__(
            data.data if isinstance(data, Tensor) else ensure_array(data),
            requires_grad=True,
            name=name,
        )


class Module:
    def __init__(self):
        self.training = True
        
    def eval(self) -> None:
        self.training = False
        for p in self.parameters():
            p.requires_grad = False
        for child in self.children():
            child.eval()
    def train(self) -> None:
        self.training = True
        for p in self.parameters():
            p.requires_grad = True
        for child in self.children():
            child.train()
    
    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def forward(self, *args) -> Tensor:
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
            
    def register_parameter(self, name: str, p: Parameter) -> None:
        assert isinstance(p, Parameter)
        vars(self)[name] = p

    def parameters(self) -> List[Parameter]:
        params = []
        for var in vars(self).items():
            if isinstance(var[1], Parameter):
                params.append(var[1])
            if isinstance(var[1], Module):
                params.extend(var[1].parameters())
        return params
    def named_parameters(self) -> Dict[str, Parameter]:
        params = {}
        for var in vars(self).items():
            if isinstance(var[1], Parameter):
                params[var[0]] = var[1]
            if isinstance(var[1], Module):
                for name, param in var[1].named_parameters().items():
                    params[var[0] + "." + name] = param
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
