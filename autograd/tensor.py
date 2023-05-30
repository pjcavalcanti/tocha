# Autograd Tensor class

# TODO:
# Maybe add @ specific for matrix multiplication

# import necessary libraries
import numpy as np
from typing import List, Callable, NamedTuple, Tuple, Union, Optional


# Define a NamedTuple to store a tensor and its gradient function
class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[["Tensor"], "Tensor"]


Arrayable = Union[float, int, list, np.ndarray]
Tensorable = Union[float, int, list, np.ndarray, "Tensor"]
Index = Union[int, slice, Tuple[Union[int, slice], ...]]


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
        name: Optional[str] = None,
    ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        if name is not None:
            self.name = name
        else:
            self.name = None

        self.shape = self.data.shape
        self.ndim = self.data.ndim
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

    def __getitem__(self, idx: Index) -> "Tensor":
        return getitem(self, idx)

    def reshape(self, shape: Tuple[int, ...]) -> "Tensor":
        return reshape(self, shape)

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        return sum(self, axis, keepdims)

    def mean(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        return mean(self, axis, keepdims)
    
    def __pow__(self, other: Union[float, int]) -> "Tensor":
        return pow(self, other)

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


def getitem(t1: Tensor, idx: Index) -> Tensor:
    data = t1.data[idx]
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.zeros_like(t1.data)
            new_grad_data[idx] = grad.data
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def reshape(t1: Tensor, shape: Tuple[int, ...]) -> Tensor:
    data = np.reshape(t1.data, shape)
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.reshape(grad.data, t1.shape)
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def transpose(t1: Tensor, axes: Tuple[int, ...]) -> Tensor:
    data = np.transpose(t1.data, axes=axes)
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.transpose(grad.data, axes)
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def neg(t1: Tensor) -> Tensor:
    data = np.negative(t1.data)
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.negative(np.multiply(grad.data, np.ones_like(grad.data)))
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn))
    return Tensor(data, requires_grad, depends_on)


def inv(t1: Tensor) -> Tensor:
    data = 1 / t1.data
    requires_grad = t1.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            new_grad_data = np.negative(np.multiply(grad.data, 1 / (t1.data**2)))
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def sum(
    t: Tensor, axis: Optional[Union[int, Tuple[int, ...]]], keepdims: bool = False
) -> Tensor:
    if axis == ():
        axis = None
    data = t.data.sum(axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: Tensor) -> Tensor:
            if keepdims:
                new_grad_data = np.multiply(grad.data, np.ones_like(t.data))
            elif axis is not None:
                new_grad_data = np.multiply(
                    np.expand_dims(grad.data, axis), np.ones_like(t.data)
                )
            else:
                new_grad_data = np.multiply(grad.data, np.ones_like(t.data))
            return Tensor(new_grad_data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)


def mean(
    t: Tensor, axis: Optional[Union[int, Tuple[int, ...]]], keepdims: bool = False
) -> Tensor:
    if axis == () or axis is None:
        size = np.prod(np.array(t.shape))
    else:
        size = np.prod(np.array(t.shape)[np.array(axis)])
    return t.sum(axis=axis, keepdims=keepdims) / size # type: ignore

def pow(t: Tensor, exponent: Union[float, int]) -> Tensor:
    data = np.power(t.data, exponent)
    requires_grad = t.requires_grad
    depends_on = []
    
    if requires_grad:
        def grad_fn(grad: Tensor):
            new_grad_data = exponent * np.multiply(grad.data, np.power(t.data, exponent - 1))
            return Tensor(new_grad_data)
    
        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)

def eq(t1: Tensor, t2: Tensor) -> bool:
    if t1.shape == t2.shape and t1.data.tolist() == t2.data.tolist():
        return True
    return False


def concatenate(ts: List[Tensor], axis: int) -> Tensor:
    data = np.concatenate([t.data for t in ts], axis=axis)
    requires_grad = any([t.requires_grad for t in ts])
    depends_on = []

    if requires_grad:
        for t in ts:
            def grad_fnt(grad: Tensor) -> Tensor:
                shape = t.shape
                
                s = [slice(None)] * t.ndim
                s[axis] = slice(0,shape[axis])
                s = tuple(s)
                new_grad_data = grad.data[s]
                return Tensor(new_grad_data)
            depends_on.append(Dependency(t, grad_fnt))

    return Tensor(data, requires_grad, depends_on)


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
            n = t2.ndim - 1
            new_grad_data = np.tensordot(
                grad.data, t2.data, axes=(tuple(range(-n, 0)), tuple(range(-n, 0)))
            )
            return Tensor(new_grad_data)

        def grad_fn2(grad: Tensor) -> Tensor:
            n = t1.ndim - 1
            new_grad_data = np.tensordot(
                t1.data, grad.data, axes=(tuple(range(0, n)), tuple(range(0, n)))
            )
            return Tensor(new_grad_data)

        depends_on.extend([Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)])

    return Tensor(data, requires_grad, depends_on)


def ndot(t1: Tensor, t2: Tensor, n: int) -> Tensor:
    data = np.tensordot(
        t1.data,
        t2.data,
        (
            tuple(range(t1.data.ndim - n, t1.data.ndim)),
            tuple(range(t2.data.ndim - n, t2.data.ndim)),
        ),
    )
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn1(grad: Tensor) -> Tensor:
            limits = (
                tuple(range(grad.data.ndim - (t2.data.ndim - n), grad.data.ndim)),
                tuple(range(0, t2.data.ndim - n)),
            )
            new_grad_data = np.tensordot(grad.data, t2.data, limits)
            return Tensor(new_grad_data)

        def grad_fn2(grad: Tensor) -> Tensor:
            limits = (
                tuple(range(0, t1.data.ndim - n)),
                tuple(range(0, t1.data.ndim - n)),
            )
            new_grad_data = np.tensordot(grad.data, t1.data, limits)
            return Tensor(new_grad_data)

        depends_on.extend([Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)])

    return Tensor(data, requires_grad, depends_on)


def tensordot(
    t1: Tensor, t2: Tensor, axes: Tuple[Tuple[int, ...], Tuple[int, ...]]
) -> Tensor:
    data = np.tensordot(t1.data, t2.data, axes)
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if requires_grad:
        original_t1_index_order = tuple(range(t1.ndim))
        original_t2_index_order = tuple(range(t2.ndim))
        contracted_t2_indices = tuple(
            i if i >= 0 else t2.ndim + i for i in axes[1]
        )  # in the order chosen for contraction
        contracted_t1_indices = tuple(
            i if i >= 0 else t1.ndim + i for i in axes[0]
        )  # in the order chosen for contraction
        non_contracted_t1_indices = tuple(
            i for i in range(t1.ndim) if i not in contracted_t1_indices
        )  # in the order of t1
        non_contracted_t2_indices = tuple(
            i for i in range(t2.ndim) if i not in contracted_t2_indices
        )  # in the order of t2

        def grad_fn1(grad: Tensor) -> Tensor:
            # first we recover the dimensions of t1 via a complementary contraction withg t2
            grad_indices_from_t2 = tuple(
                range(grad.ndim)[-len(non_contracted_t2_indices) :]
            )
            new_grad_data = np.tensordot(
                grad.data, t2.data, (grad_indices_from_t2, non_contracted_t2_indices)
            )
            # the indexes now are:
            # (non_contracted t1 indices in the order of t1, contracted t1 indices in the order of t2)
            # let's create a tuple with the current order of t1 indices:
            current_t1_index_order = non_contracted_t1_indices + tuple(
                sorted(
                    contracted_t1_indices,
                    key=lambda x: contracted_t2_indices[contracted_t1_indices.index(x)],
                )
            )
            # now we need to permute the axes of the gradient to match the order of t1
            permutation = tuple(
                current_t1_index_order.index(i) for i in original_t1_index_order
            )
            new_grad_data = np.transpose(new_grad_data, permutation)
            return Tensor(new_grad_data)

        def grad_fn2(grad: Tensor) -> Tensor:
            # complementary logic to grad_fn1, now the contraction with grad is in swapped order
            grad_indices_from_t1 = tuple(
                range(grad.ndim)[: len(non_contracted_t1_indices)]
            )
            new_grad_data = np.tensordot(
                t1.data, grad.data, (non_contracted_t1_indices, grad_indices_from_t1)
            )
            current_t2_index_order = (
                tuple(
                    sorted(
                        contracted_t2_indices,
                        key=lambda x: contracted_t1_indices[
                            contracted_t2_indices.index(x)
                        ],
                    )
                )
                + non_contracted_t2_indices
            )
            permutation = tuple(
                current_t2_index_order.index(i) for i in original_t2_index_order
            )
            new_grad_data = np.transpose(new_grad_data, permutation)
            return Tensor(new_grad_data)

        depends_on.extend([Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)])

    return Tensor(data, requires_grad, depends_on)