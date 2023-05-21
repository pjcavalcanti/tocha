from typing import Tuple
import numpy as np
from autograd.tensor import Tensor, Dependency, concatenate


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


def flatten(t1: Tensor) -> Tensor:
    data = t1.data.flatten()
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = grad.data.flatten()
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def im2col2d(im: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    k1, k2 = kernel_size[0], kernel_size[1]
    m, n = im.shape[-2:]
    s = tuple([slice(None) for _ in range(im.ndim - 2)] + [slice(0, k1), slice(0, k2)])
    cols = im[s]
    cols = cols.reshape(im.shape[:-2] + (k1 * k2, 1))
    for i in range(0, m - k1 + 1):
        for j in range(0, n - k2 + 1):
            if (i, j) == (0, 0):
                continue
            s = tuple(
                [slice(None) for _ in range(im.ndim - 2)]
                + [slice(i, i + k1), slice(j, j + k2)]
            )
            ncol = im[s]
            ncol = ncol.reshape(im.shape[:-2] + (k1 * k2, 1))
            cols = concatenate(cols, ncol, axis=-1)
    return cols


def im2row2d(im: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    k1, k2 = kernel_size[0], kernel_size[1]
    m, n = im.shape[-2:]
    s = tuple([slice(None) for _ in range(im.ndim - 2)] + [slice(0, k1), slice(0, k2)])
    cols = im[s]
    cols = cols.reshape(im.shape[:-2] + (1, k1 * k2))
    for i in range(0, m - k1 + 1):
        for j in range(0, n - k2 + 1):
            if (i, j) == (0, 0):
                continue
            s = tuple(
                [slice(None) for _ in range(im.ndim - 2)]
                + [slice(i, i + k1), slice(j, j + k2)]
            )
            ncol = im[s]
            ncol = ncol.reshape(im.shape[:-2] + (1, k1 * k2))
            cols = concatenate(cols, ncol, axis=-2)
    return cols
