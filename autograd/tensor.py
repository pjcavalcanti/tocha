# Autograd Tensor class

# import necessary libraries
import numpy as np
from typing import List, Callable, NamedTuple, Union, Optional


# Define a NamedTuple to store a tensor and its gradient function
class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


# Define a type that can be converted to a numpy array
Arrayable = Union[float, int, np.ndarray]


# Function to ensure that a given input is a numpy array
def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


# Define a class for Tensors (multi-dimensional arrays)
class Tensor:
    # Initialization of the Tensor class
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: List[Dependency] = None,
    ) -> None:
        self.data = ensure_array(data)  # make sure data is numpy array
        self.requires_grad = requires_grad  # flag for whether gradient is needed
        self.depends_on = (
            depends_on or []
        )  # list of dependencies (other tensors this one depends on)

        self.shape = self.data.shape  # shape of the tensor
        self.grad: "Tensor" = None  # gradient of the tensor

        if self.requires_grad:
            self.zero_grad()  # if gradient is required, initialize it with zeros

    # Representational function to display tensor data and its requires_grad status
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # Function to reset the gradient to zero
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    # Function to compute gradients, recursively applying the chain rule
    def backward(self, grad: "Tensor" = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            # Assuming that we only start the backward pass on the scalar with respect to which we want to differentiate,
            # we can say that if grad is None, then we are differentiating with respect to self, so grad = 1
            if self.shape == ():
                grad = Tensor(1.0)  # if tensor is a scalar, start with grad of 1
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        # If we already have a gradient, accumulate the new gradient
        self.grad.data += grad.data  # increment gradient by grad

        for dependency in self.depends_on:
            # Compute the gradient of the dependency tensor
            # grad_fn(grad) = grad * d(self.data)/d(dependency.data)
            #               = d(loss)/d(self.data) * d(self.data)/d(dependency.data)
            #               = d(loss)/d(dependency.data)
            backward_grad = dependency.grad_fn(
                grad.data
            )  # compute gradient for dependencies
            dependency.tensor.backward(
                Tensor(backward_grad)
            )  # recursively call backward

    # Function to compute the sum of tensor elements
    def sum(self) -> "Tensor":
        return tensor_sum(self)


# Function to create a new tensor that is the sum of all elements in a given tensor
def tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()  # compute sum of all elements
    requires_grad = t.requires_grad

    if requires_grad:
        # define a gradient function that broadcasts the incoming gradient to the shape of t
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # grad_fn(grad) = grad * d(t.data.sum())/d(t.data)
            #               = grad * ones_like(t.data)
            # since t.data.sum() is a scalar, d(t.data.sum())/d(t.data) = has the same shape as t.data
            # that if T = T_XYZ, and U = U_IJK, then d(T)/d(U) = V_xyzijk = d(T_xyz)/d(U_ijk)
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]  # new tensor depends on t
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)  # return new tensor
