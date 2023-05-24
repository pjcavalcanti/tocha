import unittest
import numpy as np
import tocha
import torch


class TestTensorMul(unittest.TestCase):
    def test_mul_simple(self):
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)
        t2 = tocha.tensor([4, 5, 6], requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [1 * 4, 2 * 5, 3 * 6]

    def test_mul_backward_simple(self):
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)
        t2 = tocha.tensor([4, 5, 6], requires_grad=True)

        t3 = t1 * t2

        t3.backward(tocha.tensor([2, 3, 4]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [2 * 4, 3 * 5, 4 * 6]  # grad*t2
        assert t2.grad.data.tolist() == [2 * 1, 3 * 2, 4 * 3]  # grad*t1

    def test_mul_backward_broadcast1(self):
        t1 = tocha.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
        t2 = tocha.tensor([7, 8, 9], requires_grad=True)  # (3,) -> (2,3)

        t3 = t1 * t2

        t3.backward(tocha.tensor([[10, 11, 12], [13, 14, 15]]))

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
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)
        t2 = tocha.tensor(4, requires_grad=True)

        t3 = t1 * t2

        t3.backward(tocha.tensor([5, 6, 7]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [20, 24, 28]
        assert t2.grad.data.tolist() == 38
    def test_compare_torch(self):
        np.random.seed(0)
        options = [
            (np.random.randn(2,3,4,5), np.random.randn(2,3,4,5)),
            (np.random.randn(2,3,4,5), np.random.randn(3,4,5)),
            (np.random.randn(2,3,4,5), np.random.randn(4,5)),
            (np.random.randn(2,3,4,5), np.random.randn(1,3,4,5)),
            (np.random.randn(2,3,4,5), np.random.randn(1,1,4,5)),
        ]
        for i in range(len(options)):
            anp, bnp = options[i]
            a = tocha.tensor(anp, requires_grad=True)
            b = tocha.tensor(bnp, requires_grad=True)
            a_torch = torch.tensor(anp, requires_grad=True)
            b_torch = torch.tensor(bnp, requires_grad=True)
            
            c =  a * b
            c_torch = a_torch * b_torch
            
            grad = np.random.randn(*c.data.shape)
            grad_torch = torch.from_numpy(grad)
            grad = tocha.tensor(grad)
            
            print(type(grad), type(grad_torch))
            
            c.backward(grad)
            c_torch.backward(grad_torch)
            
            assert np.allclose(c.data, c_torch.data.numpy())
            assert np.allclose(a.grad.data, a_torch.grad.numpy()) # type: ignore