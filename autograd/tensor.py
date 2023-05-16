# Autograd Tensor class

# import necessary libraries
import numpy as np
from typing import List, Callable, NamedTuple, Union, Optional


# Define a NamedTuple to store a tensor and its gradient function
class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[["Tensor"], "Tensor"]


# Define a type that can be converted to a numpy array
Arrayable = Union[float, int, list, np.ndarray]


# Function to ensure that a given input is a numpy array
def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union[Arrayable, "Tensor"]


def ensure_tensor(tensorable: Arrayable) -> "Tensor":
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


# Define a class for Tensors (multi-dimensional arrays)
class Tensor:
    # Initialization of the Tensor class
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: Optional[List[Dependency]] = None,
    ) -> None:
        self.data = ensure_array(data)  # make sure data is numpy array
        self.requires_grad = requires_grad  # flag for whether gradient is needed
        self.depends_on = (
            depends_on or []
        )  # list of dependencies (other tensors this one depends on)

        self.shape = self.data.shape  # shape of the tensor
        self.grad: Optional["Tensor"] = None  # gradient of the tensor

        if self.requires_grad:
            self.zero_grad()  # if gradient is required, initialize it with zeros

    # Representational function to display tensor data and its requires_grad status
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # Function to reset the gradient to zero
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    # Function to compute gradients, recursively applying the chain rule
    def backward(self, grad: Optional["Tensor"] = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            # Assuming that we only start the backward pass on the scalar with respect to which we want to differentiate,
            # we can say that if grad is None, then we are differentiating with respect to self, so grad = 1
            if self.shape == ():
                grad = Tensor(1.0)  # if tensor is a scalar, start with grad of 1
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        # If we already have a gradient, accumulate the new gradient
        if self.grad is not None:
            self.grad.data = self.grad.data + grad.data  # increment gradient by grad

        for dependency in self.depends_on:
            # Compute the gradient of the dependency tensor
            # grad_fn(grad) = grad * d(self.data)/d(dependency.data)
            #               = d(loss)/d(self.data) * d(self.data)/d(dependency.data)
            #               = d(loss)/d(dependency.data)
            backward_grad = dependency.grad_fn(
                grad
            )  # compute gradient for dependencies
            dependency.tensor.backward(backward_grad)  # recursively call backward

    # Function to compute the sum of tensor elements
    def sum(self) -> "Tensor":
        return tensor_sum(self)

    def __add__(self, other: "Tensor") -> "Tensor":
        return add(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return mul(self, other)


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn1(grad: Tensor) -> Tensor:
            new_grad_data = grad.data * t2.data

            dims_added = len(grad.data.shape) - len(t1.data.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for index, dimension in enumerate(t1.data.shape):
                if dimension == 1:
                    new_grad_data = new_grad_data.sum(axis=index, keepdims=True)

            return Tensor(new_grad_data)

        depends_on.append(Dependency(t1, grad_fn1))

        def grad_fn2(grad: Tensor) -> Tensor:
            new_grad_data = grad.data * t1.data

            dims_added = len(grad.data.shape) - len(t2.data.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for index, dimension in enumerate(t2.data.shape):
                if dimension == 1:
                    new_grad_data = new_grad_data.sum(axis=index, keepdims=True)

            return Tensor(new_grad_data)

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


# Function to create a new tensor that is the sum of all elements in a given tensor
def tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()  # compute sum of all elements
    requires_grad = t.requires_grad

    if requires_grad:
        # define a gradient function that broadcasts the incoming gradient to the shape of t
        def grad_fn(grad: Tensor) -> Tensor:
            # grad_fn(grad) = grad * d(t.data.sum())/d(t.data)
            #               = grad * ones_like(t.data)
            # since t.data.sum() is a scalar, d(t.data.sum())/d(t.data) = has the same shape as t.data
            # that if T = T_XYZ, and U = U_IJK, then d(T)/d(U) = V_xyzijk = d(T_xyz)/d(U_ijk)
            return Tensor(grad.data * np.ones_like(t.data))

        depends_on = [Dependency(t, grad_fn)]  # new tensor depends on t
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)  # return new tensor


def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    if requires_grad:

        def grad_fn1(grad: Tensor) -> Tensor:
            # To understand the grads, model the broadcast with tensors as follows:
            #
            # T_XY + U_Y = T_XY + 1_X U_Y, where 1_X has all entries 1_x = 1
            # T_XY + U_X'Y = T_XY + 1_XX' U_X'Y, where 1_XX' has all entries 1_xx' = 1
            #                                          and dim(X') = 1 (einsum assumed)
            #
            # With this, one can calculate the grads with the usual chain rule
            new_grad_data = grad.data

            dims_added = len(grad.data.shape) - len(t1.data.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    new_grad_data = new_grad_data.sum(axis=i, keepdims=True)
            return Tensor(new_grad_data * np.ones_like(t1.data))

        def grad_fn2(grad: Tensor) -> Tensor:
            new_grad_data = grad.data
            dims_added = len(grad.data.shape) - len(t2.data.shape)
            for _ in range(dims_added):
                new_grad_data = new_grad_data.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    new_grad_data = new_grad_data.sum(axis=i, keepdims=True)
            return Tensor(new_grad_data * np.ones_like(t2.data))

        depends_on = [Dependency(t1, grad_fn1), Dependency(t2, grad_fn2)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)
