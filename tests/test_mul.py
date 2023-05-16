import unittest
import numpy as np
from autograd.tensor import Tensor


class TestTensorMul(unittest.TestCase):
    def test_mul_simple(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [1 * 4, 2 * 5, 3 * 6]

    def test_mul_backward_simple(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 * t2

        t3.backward(Tensor([2, 3, 4]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [2 * 4, 3 * 5, 4 * 6]  # grad*t2
        assert t2.grad.data.tolist() == [2 * 1, 3 * 2, 4 * 3]  # grad*t1

    def test_mul_backward_broadcast1(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
        t2 = Tensor([7, 8, 9], requires_grad=True)  # (3,) -> (2,3)

        t3 = t1 * t2

        t3.backward(Tensor([[10, 11, 12], [13, 14, 15]]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [
            [10 * 7, 11 * 8, 12 * 9],
            [13 * 7, 14 * 8, 15 * 9],
        ]
        assert t2.grad.data.tolist() == [
            10 * 1 + 13 * 4,
            11 * 2 + 14 * 5,
            12 * 3 + 15 * 6,
        ]

    def test_mul_scalar(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor(4, requires_grad=True)

        t3 = t1 * t2

        t3.backward(Tensor([5, 6, 7]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [20, 24, 28]
        assert t2.grad.data.tolist() == 38
