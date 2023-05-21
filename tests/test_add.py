import unittest

import tocha


class TestTensorAdd(unittest.TestCase):
    def test_sum_simple(self):
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)
        t2 = tocha.tensor([4, 5, 6], requires_grad=True)

        t3 = t1 + t2

        t3.backward(tocha.tensor([7, 8, 9]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [7, 8, 9]
        assert t2.grad.data.tolist() == [7, 8, 9]

    def test_sum_broadcast1(self):
        t1 = tocha.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
        t2 = tocha.tensor([7, 8, 9], requires_grad=True)  # (3,) -> (2,3)

        t3 = t1 + t2

        t3.backward(tocha.tensor([[10, 11, 12], [13, 14, 15]]))
        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [[10, 11, 12], [13, 14, 15]]
        assert t2.grad.data.tolist() == [10 + 13, 11 + 14, 12 + 15]

    def test_sum_broadcast2(self):
        t1 = tocha.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
        t2 = tocha.tensor([[7, 8, 9]], requires_grad=True)  # (3,) -> (2,3)

        t3 = t1 + t2

        t3.backward(tocha.tensor([[10, 11, 12], [13, 14, 15]]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [[10, 11, 12], [13, 14, 15]]
        assert t2.grad.data.tolist() == [[10 + 13, 11 + 14, 12 + 15]]

    def test_sum_commutativity_associativity(self):
        t1 = tocha.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
        t2 = tocha.tensor([[7, 8, 9]], requires_grad=True)  # (3,) -> (2,3)
        t3 = tocha.tensor([[10, 11, 12], [13, 14, 15]], requires_grad=True)  # (2,3)

        t4 = t1 + t2 + t3
        t5 = (t1 + t2) + t3
        t6 = t1 + (t2 + t3)
        t7 = t2 + t1 + t3
        t8 = t2 + t3 + t1

        t4.backward(tocha.tensor([[1, 1, 1], [1, 1, 1]]))
        t5.backward(tocha.tensor([[1, 1, 1], [1, 1, 1]]))
        t6.backward(tocha.tensor([[1, 1, 1], [1, 1, 1]]))
        t7.backward(tocha.tensor([[1, 1, 1], [1, 1, 1]]))
        t8.backward(tocha.tensor([[1, 1, 1], [1, 1, 1]]))

        assert (
            t4.data.tolist()
            == t5.data.tolist()
            == t6.data.tolist()
            == t7.data.tolist()
            == t8.data.tolist()
        )
