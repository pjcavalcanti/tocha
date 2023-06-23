from typing import Iterable, Tuple, Optional

from tocha.module import Parameter
from tocha import Tensor
import numpy as np


class Optimizer:
    def __init__(self) -> None:
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:  # type: ignore
            p.zero_grad()


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
    ) -> None:
        assert lr > 0
        assert momentum >= 0 and momentum <= 1
        if nesterov is True:
            assert dampening == 0
        self.first_step = True
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.lr = lr
        self.params = list(params)
        self.v = [np.zeros(p.shape).astype(p.data.dtype) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            # In Sutskever et al. we have differently, but I will mirror pytorch

            g_t = p.grad.data  # type: ignore

            if self.weight_decay != 0:
                g_t = g_t + self.weight_decay * p.data

            if self.first_step is True:
                self.v[i] = self.momentum * self.v[i] + g_t  # type: ignore
            else:
                self.v[i] = self.momentum * self.v[i] + (1 - self.dampening) * g_t  # type: ignore

            if self.nesterov:
                p.data = p.data - self.lr * (self.momentum * self.v[i] + g_t)  # type: ignore
            else:
                p.data = p.data - self.lr * self.v[i]

        self.first_step = False


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        betas: Tuple[float, float],
        eps: float = 1e-8,
        weight_decay: Optional[float] = None,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = (
            weight_decay if (weight_decay is not None and weight_decay != 0) else None
        )

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        if self.weight_decay is None:
            for i, p in enumerate(self.params):
                g_t = p.grad.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g_t * g_t)
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                p.data += -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        else:
            for i, p in enumerate(self.params):
                g_t = p.grad.data + self.weight_decay * p.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g_t**2
                p.data += (
                    -self.lr
                    * (self.m[i] / (1 - self.beta1**self.t))
                    / (np.sqrt(self.v[i] / (1 - self.beta2**self.t)) + self.eps)
                )
