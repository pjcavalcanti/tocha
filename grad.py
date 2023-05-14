import torch

class Tensor:
    """stores a tensor and its gradient"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def backward(self):
        topological_order = []
        visited = set()
        def build_topological_order(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_topological_order(child)
                topological_order.append(tensor)
        build_topological_order(self)

        self.grad = torch.ones_like(self.data)
        for tensor in reversed(topological_order):
            tensor._backward()

    def relu(self):
        out = Tensor(self.data.clamp(min=0), (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0).float() * out.grad
        out._backward = _backward

        return out

    def einsum(self, other, equation):
        out = Tensor(torch.einsum(equation, self.data, other.data), (self, other), 'einsum')

        def _backward(): # i need to parse the equation to get the indices
            
            self.grad += torch.einsum(equation, out.grad, other.data)
            other.grad += torch.einsum(equation, self.data, out.grad)


    # def __matmul__(self, other):
    #     out = Tensor(self.data @ other.data, (self, other), '@')
        
    #     def _backward():
    #         self.grad += out.grad @ other.data.t()
    #         other.grad += self.data.t() @ out.grad
    #     out._backward = _backward

        return out
    
    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad # d(out)/d(self) = 1
            other.grad += out.grad # d(out)/d(other) = 1
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            # at call, the 
            self.grad += other.data * out.grad # d(out)/d(self) = other
            other.grad += self.data * out.grad # d(out)/d(other) = self
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        out = Tensor(self.data ** other.data, (self, other), '**')
        
        def _backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad # d(out)/d(self) = other * self**(other-1)
            other.grad += (self.data ** other.data * torch.log(self.data)) * out.grad # d(out)/d(other) = self**other * log(self)

        out._backward = _backward
        
        return out
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"