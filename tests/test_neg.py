import unittest

import tocha


class TestTensorNegate(unittest.TestCase):
    def test_negate_tensor(self):
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)

        t2 = -t1

        assert t2.data.tolist() == [-1, -2, -3]

    def test_negate_tensor_backward(self):
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)

        t2 = -t1
        t2.backward(tocha.tensor([4, 5, 6]))

        assert t1.grad is not None
        assert t1.grad.data.tolist() == [-4, -5, -6]

    def test_negate_scalar(self):
        t1 = tocha.tensor(4, requires_grad=True)

        t2 = -t1

        assert t2.data.tolist() == -4

    def test_negate_scalar_backward(self):
        t1 = tocha.tensor(4, requires_grad=True)

        t2 = -t1
        t2.backward(tocha.tensor(7))

        assert t1.grad is not None
        assert t1.grad.data.tolist() == -7
