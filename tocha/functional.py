import numpy as np
from autograd.tensor import Tensor, Dependency


def relu(t: Tensor) -> Tensor:
    data = np.maximum(t.data, 0)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, np.greater(t.data, 0).astype(int))
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def sigmoid(t: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-t.data))
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, np.multiply(data, 1 - data))
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def tanh(t: Tensor) -> Tensor:
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, 1 - np.square(data))
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)
