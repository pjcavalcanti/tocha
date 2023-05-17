# Autograd Tensor class

# TODO:
# Maybe add @ specific for matrix multiplication

# import necessary libraries
import numpy as np
from typing import List, Callable, NamedTuple, Union, Optional


# Define a NamedTuple to store a tensor and its gradient function
class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[["Tensor"], "Tensor"]


Arrayable = Union[float, int, list, np.ndarray]
Tensorable = Union[Arrayable, "Tensor"]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


def ensure_tensor(tensorable: Tensorable) -> "Tensor":
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: Optional[List[Dependency]] = None,
    ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []

        self.shape = self.data.shape
        self.ndims = self.data.ndim
        self.grad: Optional["Tensor"] = None

        if self.requires_grad:
            self.zero_grad()  # if gradient is required, initialize it with zeros

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    # Function to compute gradients, recursively applying the chain rule
    def backward(self, grad: Optional["Tensor"] = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            # Assuming that we only start the backward pass on the scalar with respect to which we want to differentiate,
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        # If we already have a gradient, accumulate the new gradient
        if self.grad is not None:
            self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad)
            dependency.tensor.backward(backward_grad)

    # Function to compute the sum of tensor elements

    def sum(self) -> "Tensor":
        return tensor_sum(self)

    def __len__(self) -> int:
        return len(self.data)

    def __neg__(self) -> "Tensor":
        return neg(self)

    def __truediv__(self, other: Tensorable) -> "Tensor":
        return divide(self, ensure_tensor(other))

    def __rtruediv__(self, other: Tensorable) -> "Tensor":
        return divide(ensure_tensor(other), self)  # not commutative

    def __eq__(self, other: "Tensor") -> bool:
        return eq(self, other)

    def __add__(self, other: Tensorable) -> "Tensor":
        return add(self, ensure_tensor(other))

    def __radd__(self, other: Tensorable) -> "Tensor":
        return add(self, ensure_tensor(other))

    def __sub__(self, other: Tensorable) -> "Tensor":
        return sub(self, ensure_tensor(other))

    def __rsub__(self, other: Tensorable) -> "Tensor":
        return sub(self, ensure_tensor(other))

    def __mul__(self, other: Tensorable) -> "Tensor":
        return mul(self, ensure_tensor(other))

    def __rmul__(self, other: Tensorable) -> "Tensor":
        return mul(self, ensure_tensor(other))

    def __matmul__(self, other: Tensorable) -> "Tensor":
        return matmultensor(self, ensure_tensor(other))

    def __rmatmul__(self, other: Tensorable) -> "Tensor":
        return matmultensor(ensure_tensor(other), self)  # not commutative


def tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            return Tensor(np.multiply(grad.data, np.ones_like(t.data)))

        depends_on.extend([Dependency(t, grad_fn)])

    return Tensor(data, requires_grad, depends_on)


def neg(t1: Tensor) -> Tensor:
    data = np.negative(t1.data)
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.negative(np.multiply(grad.data, np.ones_like(grad)))
            return Tensor(new_grad_data)

        depends_on.extend([Dependency(t1, grad_fn)])
    return Tensor(data, requires_grad, depends_on)


def inv(t1: Tensor) -> Tensor:
    data = 1 / t1.data
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.negative(np.multiply(grad.data, 1 / (t1.data**2)))
            return Tensor(new_grad_data)

        depends_on.extend([Dependency(t1, grad_fn)])

    return Tensor(data, requires_grad, depends_on)


def eq(t1: Tensor, t2: Tensor):
    if t1.shape == t2.shape and t1.data.tolist() == t2.data.tolist():
        return True
    return False


def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.add(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn1(grad: Tensor) -> Tensor:
            new_grad_data = grad.data

            dims_added = len(grad.shape) - len(t1.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    new_grad_data = new_grad_data.sum(axis=i, keepdims=True)
            return Tensor(np.multiply(new_grad_data, np.ones_like(t1.data)))

        def grad_fn2(grad: Tensor) -> Tensor:
            new_grad_data = grad.data
            dims_added = len(grad.shape) - len(t2.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    new_grad_data = new_grad_data.sum(axis=i, keepdims=True)
            return Tensor(np.multiply(new_grad_data, np.ones_like(t2.data)))

        depends_on.extend([Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)])

    return Tensor(data, requires_grad, depends_on)


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.subtract(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn1(grad: Tensor) -> Tensor:
            new_grad_data = grad.data

            dims_added = len(grad.shape) - len(t1.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    new_grad_data = new_grad_data.sum(axis=i, keepdims=True)
            return Tensor(np.multiply(new_grad_data, np.ones_like(t1.data)))

        def grad_fn2(grad: Tensor) -> Tensor:
            new_grad_data = np.negative(grad.data)
            dims_added = len(grad.shape) - len(t2.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    new_grad_data = new_grad_data.sum(axis=i, keepdims=True)
            return Tensor(np.multiply(new_grad_data, np.ones_like(t2.data)))

        depends_on.extend([Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)])
    return Tensor(data, requires_grad, depends_on)


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.multiply(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn1(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, t2.data)

            dims_added = len(grad.shape) - len(t1.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for index, dimension in enumerate(t1.shape):
                if dimension == 1:
                    new_grad_data = new_grad_data.sum(axis=index, keepdims=True)

            return Tensor(new_grad_data)

        def grad_fn2(grad: Tensor) -> Tensor:
            new_grad_data = np.multiply(grad.data, t1.data)

            dims_added = len(grad.shape) - len(t2.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for index, dimension in enumerate(t2.shape):
                if dimension == 1:
                    new_grad_data = new_grad_data.sum(axis=index, keepdims=True)

            return Tensor(new_grad_data)

        depends_on.extend([Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)])

    return Tensor(data, requires_grad, depends_on)


def divide(t1: Tensor, t2: Tensor) -> Tensor:
    # using inv, we don't need to deal with broadcasting again
    # because mul deals with broadcasting already
    return mul(t1, inv(t2))


def matmultensor(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.tensordot(t1.data, t2.data, (t1.data.ndim - 1, 0))
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn1(grad: Tensor) -> Tensor:
            n = t2.ndims - 1
            new_grad_data = np.tensordot(
                grad.data, t2.data, axes=(tuple(range(-n, 0)), tuple(range(-n, 0)))
            )
            return Tensor(new_grad_data)

        def grad_fn2(grad: Tensor) -> Tensor:
            n = t1.ndims - 1
            new_grad_data = np.tensordot(
                t1.data, grad.data, axes=(tuple(range(0, n)), tuple(range(0, n)))
            )
            return Tensor(new_grad_data)

        depends_on.extend([Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)])

    return Tensor(data, requires_grad, depends_on)
