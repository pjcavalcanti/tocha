from typing import Tuple, Union
import numpy as np
from autograd.tensor import Tensor, Dependency, concatenate


def relu(t: Tensor) -> Tensor:
    # TODO make unit tests
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
    # TODO make unit tests
    data = 1 / (1 + np.exp(-t.data))
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, np.multiply(data, 1 - data))
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def exp(t: Tensor) -> Tensor:
    # TODO make unit tests
    data = np.exp(t.data)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, data)
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def softmax(t1: Tensor, dim: Union[int, Tuple[int,...]]) -> Tensor:
    # TODO make unit tests
    t2 = exp(t1)
    t2 = t2 / t2.sum(axis=dim, keepdims=True)
    return t2

def cross_entropy(x: Tensor, y: Tensor) -> Tensor:
    # TODO make unit tests
    out = softmax(x, dim=1)
    return - log(out[np.arange(0, out.shape[0]), y.data]).mean() # type: ignore

def log(t: Tensor, epsilon = 1e-8) -> Tensor:
    data = np.log(t.data + epsilon)
    requires_grad = t.requires_grad
    depends_on = []
    
    if requires_grad:
        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, 1 / (t.data + epsilon))
            return Tensor(new_grad_data)
        
        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)

def tanh(t: Tensor) -> Tensor:
    # TODO make unit tests
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
    # TODO make unit tests
    data = t1.data.flatten()
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = grad.data.flatten()
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def _im2col_np(t: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    # TODO: add stride and padding
    assert len(t.shape) == 4, "tensor must be a 4D array for im2col"
    # Get paramaters
    k1, k2 = kernel_size[0], kernel_size[1]
    batch, c_in, height, width = t.shape
    # Initialize matrix for submatrices
    blocks = np.zeros((batch, c_in, k1 * k2, (height - k1 + 1) * (width - k2 + 1)))
    col = 0
    for i in range(0, height - k1 + 1):
        for j in range(0, width - k2 + 1):
            # Get submatrix
            block = t[:, :, i : i + k1, j : j + k2].reshape(batch, c_in, k1 * k2)
            blocks[:, :, :, col] = block
            col += 1
    return blocks


def _col2im_np(
    t1: np.ndarray, kernel_size: Tuple[int, int], original_size: Tuple[int, int]
) -> np.ndarray:
    b, c, hp, wp = t1.shape
    k1, k2 = kernel_size[0], kernel_size[1]
    height = original_size[0]
    width = original_size[1]
    t2 = np.zeros((b, c, height, width))
    col = 0
    for i in range(0, height - k1 + 1):
        for j in range(0, width - k1 + 1):
            # Pick a grad block from a column
            block = t1[:, :, :, col].reshape(b, c, k1, k2)
            # Increment the grad
            t2[:, :, i : i + k1, j : j + k2] += block
            col += 1
    return t2


def im2col(t: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    data = _im2col_np(t.data, kernel_size)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = _col2im_np(grad.data, kernel_size, t.shape[-2:])
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def im2col_slow(im: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    # TODO: add stride and padding
    # Get paramaters
    k1, k2 = kernel_size[0], kernel_size[1]
    m, n = im.shape[-2:]
    # Get first submatrix as a column
    s = tuple([slice(None) for _ in range(im.ndim - 2)] + [slice(0, k1), slice(0, k2)])
    cols = im[s]
    cols = cols.reshape(im.shape[:-2] + (k1 * k2, 1))
    # Concatenate the rest of the submatrices as columns
    for i in range(0, m - k1 + 1):
        for j in range(0, n - k2 + 1):
            if (i, j) == (0, 0):
                continue
            # Get submatrix
            s = tuple(
                [slice(None) for _ in range(im.ndim - 2)]
                + [slice(i, i + k1), slice(j, j + k2)]
            )
            ncol = im[s]
            # Reshape to column
            ncol = ncol.reshape(im.shape[:-2] + (k1 * k2, 1))
            # Concatenate
            cols = concatenate(cols, ncol, axis=-1)
    return cols


def im2row_slow(im: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    # TODO: add stride and padding
    # Get paramaters
    k1, k2 = kernel_size[0], kernel_size[1]
    m, n = im.shape[-2:]
    # Get first submatrix as a row
    s = tuple([slice(None) for _ in range(im.ndim - 2)] + [slice(0, k1), slice(0, k2)])
    cols = im[s]
    cols = cols.reshape(im.shape[:-2] + (1, k1 * k2))
    # Concatenate the rest of the submatrices as rows
    for i in range(0, m - k1 + 1):
        for j in range(0, n - k2 + 1):
            if (i, j) == (0, 0):
                continue
            # Get submatrix
            s = tuple(
                [slice(None) for _ in range(im.ndim - 2)]
                + [slice(i, i + k1), slice(j, j + k2)]
            )
            ncol = im[s]
            # Reshape to row
            ncol = ncol.reshape(im.shape[:-2] + (1, k1 * k2))
            # Concatenate
            cols = concatenate(cols, ncol, axis=-2)
    return cols
