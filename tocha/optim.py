from typing import Iterable
from tocha.module import Parameter
from tocha import Tensor
import numpy as np

class Optimizer:
    def __init__(self) -> None:
        raise NotImplementedError
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
    
# class SGD(Optimizer):
#     def __init__(self, params: Iterable[Parameter], lr: float, momentum: float = 0) -> None:
#         assert lr > 0
#         assert momentum >= 0 and momentum <= 1
#         self.params = params
#         self.lr = lr
#         self.momentum = momentum
#         self.v = [np.zeros(p.shape) for p in self.params]

#     def step(self):
#         for i, p in enumerate(self.params):
#             self.v[i] = self.momentum * self.v[i] - self.lr * p.grad.data
#             p.data += self.v[i] 