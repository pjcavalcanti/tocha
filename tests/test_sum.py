import unittest

import tocha


class TestTensorSum(unittest.TestCase):
    def test_sum(self):
        t1 = tocha.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t2 = t1.sum()

        t2.backward()

        assert t1.grad is not None
        assert t1.grad.data.tolist() == [1.0, 1.0, 1.0]

    def test_sum_with_grad(self):
        t1 = tocha.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t2 = t1.sum()

        t2.backward(tocha.tensor(3))

        assert t1.grad is not None
        assert t1.grad.data.tolist() == [1.0 * 3, 1.0 * 3, 1.0 * 3]
