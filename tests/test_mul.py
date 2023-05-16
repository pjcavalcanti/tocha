import unittest

from autograd.tensor import Tensor


class TestSum(unittest.TestCase):
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

        assert t1.grad.data.tolist() == [1 * 2, 2 * 3, 3 * 4]

    # def test_sum_broadcast1(self):
    #     t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    #     t2 = Tensor([7, 8, 9], requires_grad=True)  # (3,) -> (2,3)

    #     t3 = t1 + t2

    #     t3.backward(Tensor([[10, 11, 12], [13, 14, 15]]))

    #     assert t1.grad.data.tolist() == [[10, 11, 12], [13, 14, 15]]
    #     assert t2.grad.data.tolist() == [10 + 13, 11 + 14, 12 + 15]

    # def test_sum_broadcast2(self):
    #     t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    #     t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (3,) -> (2,3)

    #     t3 = t1 + t2

    #     t3.backward(Tensor([[10, 11, 12], [13, 14, 15]]))

    #     assert t1.grad.data.tolist() == [[10, 11, 12], [13, 14, 15]]
    #     assert t2.grad.data.tolist() == [[10 + 13, 11 + 14, 12 + 15]]

    # def test_sum_commutativity_associativity(self):
    #     t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    #     t2 = Tensor([[7, 8, 9]], requires_grad=True)  # (3,) -> (2,3)
    #     t3 = Tensor([[10, 11, 12], [13, 14, 15]], requires_grad=True)  # (2,3)

    #     t4 = t1 + t2 + t3
    #     t5 = (t1 + t2) + t3
    #     t6 = t1 + (t2 + t3)
    #     t7 = t2 + t1 + t3
    #     t8 = t2 + t3 + t1

    #     t4.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
    #     t5.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
    #     t6.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
    #     t7.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
    #     t8.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

    #     assert (
    #         t4.data.tolist()
    #         == t5.data.tolist()
    #         == t6.data.tolist()
    #         == t7.data.tolist()
    #         == t8.data.tolist()
    #     )
