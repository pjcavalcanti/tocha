import unittest

from autograd.tensor import Tensor


class TestTensorInPlaceOperations(unittest.TestCase):
    def test_inplace_add1(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t1 = t1 + 3
        t1 = t1.sum()
        t1.backward()

        assert t1.grad is not None
        assert t1.grad.data.tolist() == 1
        assert t1.grad.data.shape == ()

    def test_inplace_add2(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1 + 3
        t2 = t2.sum()
        t2.backward()

        assert t1.grad is not None
        assert t1.grad.data.tolist() == [1, 1, 1]
        assert t1.grad.data.shape == (3,)

    def test_inplace_mul1(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t1 = t1 * 3
        t1 = t1.sum()
        t1.backward()

        assert t1.grad is not None
        assert t1.grad.data.tolist() == 1
        assert t1.grad.data.shape == ()

    def test_inplace_mul2(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1 * 3
        t2 = t2.sum()
        t2.backward()

        assert t1.grad is not None
        assert t1.grad.data.tolist() == [3, 3, 3]
        assert t1.grad.data.shape == (3,)

    def test_inplace_sequence(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1 + 3
        t3 = t2 * 2
        t3 = t3.sum()
        t3.backward()

        assert t1.grad is not None
        assert t1.grad.data.tolist() == [2, 2, 2]
        assert t1.grad.data.shape == (3,)

    def test_inplace_with_other_tensors(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 + t2
        t3 = t3 * t2
        t3 = t3.sum()
        t3.backward()

        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        d = c * b
        e = d.sum()
        e.backward()

        assert a.grad is not None
        assert b.grad is not None
        assert e.grad is not None

        assert t1.grad is not None
        assert t1.grad.data.tolist() == a.grad.data.tolist()
        assert t1.grad.data.shape == (3,)

        assert t2.grad is not None
        assert t2.grad.data.tolist() == b.grad.data.tolist()
        assert t2.grad.data.shape == (3,)

        assert t3.grad is not None
        assert t3.grad.data.tolist() == e.grad.data.tolist()
