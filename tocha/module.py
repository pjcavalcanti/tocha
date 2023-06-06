from typing import Any, Iterator, List, Tuple, Union, Dict, Optional
from autograd.tensor import Tensor, Arrayable, ensure_array


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor], name: Optional[str] = None):
        super().__init__(
            data.data if isinstance(data, Tensor) else ensure_array(data),
            requires_grad=True,
            name=name,
        )
        
    def __repr__(self):
        return f"Parameter({self.data}, requires_grad={self.requires_grad})"


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
    
    def __call__(self, *args: Any, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args) -> Tensor:
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
            
    def register_parameter(self, name: str, p: Parameter) -> None:
        assert isinstance(p, Parameter)
        vars(self)[name] = p
        
    def register_module(self, name: str, module: "Module") -> None:
        assert isinstance(module, Module)
        vars(self)[name] = module

    def parameters(self) -> Iterator[Parameter]:
        for var in vars(self).items():
            if isinstance(var[1], Parameter):
                yield var[1]
            if isinstance(var[1], Module):
                for param in var[1].parameters():
                    yield param
                yield from var[1].parameters()
                
    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        params = {}
        for var in vars(self).items():
            if isinstance(var[1], Parameter):
                yield var[0], var[1]
            if isinstance(var[1], Module):
                for name, param in var[1].named_parameters():
                    yield f"{var[0]}.{name}", param
                    
    def children(self) -> Iterator["Module"]:
        for var in vars(self).items():
            if isinstance(var[1], Module):
                yield var[1]
            


class ParameterList(Module):
    def __init__(self, parameters: List[Parameter]):
        self.parameterlist = parameters

    def parameters(self) -> List[Parameter]:
        return self.parameterlist
