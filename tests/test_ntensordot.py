import unittest
import pytest

import numpy as np

from autograd.tensor import Tensor, ndot


class TestTensorNDot(unittest.TestCase):
    def test_simple_ndot_as_matmul(self):
        # matmul test
        t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)  # (3,2)
        t2 = Tensor([[10], [20]], requires_grad=True)  # (2,1)
        t3 = t1 @ t2

        # matmul via ndot
        T1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        T2 = Tensor(np.array([[10], [20]]).transpose(), requires_grad=True)
        T3 = ndot(T1, T2, 1)

        assert t3.data.tolist() == [[50], [110], [170]]
        assert T3.data.tolist() == [[50], [110], [170]]

        grad = Tensor([[-1], [-2], [-3]])
        t3.backward(grad)
        T3.backward(grad)

        # the matmul grads work
        np.testing.assert_array_equal(t1.grad.data, grad.data @ t2.data.T)  # type: ignore
        np.testing.assert_array_equal(t2.grad.data, t1.data.T @ grad.data)  # type: ignore
        # the ndot reproduces matmul
        np.testing.assert_array_equal(T2.grad.data.transpose(), t2.grad.data)  # type: ignore
        np.testing.assert_array_equal(T1.grad.data, t1.grad.data)  # type: ignore

    def test_simple_hilbert_schmidt_ndot(self):
        t1 = Tensor([[1, 2], [3, 4]], requires_grad=True)  # (2,2)
        t2 = Tensor([[1, 2], [3, 4]], requires_grad=True)  # (2,2)
        t3 = ndot(t1, t2, 2)

        assert t3.data.tolist() == 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4
