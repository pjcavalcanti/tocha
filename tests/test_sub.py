import unittest
import tocha


class TestTensorSub(unittest.TestCase):
    def test_sub_two_tensors(self):
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)
        t2 = tocha.tensor([4, 5, 6], requires_grad=True)

        t3 = t1 - t2

        t3.backward(tocha.tensor([7, 8, 9]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [7, 8, 9]
        assert t2.grad.data.tolist() == [-7, -8, -9]

    def test_sub_tensor_scalar(self):
        t1 = tocha.tensor([1, 2, 3], requires_grad=True)
        t2 = tocha.tensor(4, requires_grad=True)

        t3 = t1 - t2

        t3.backward(tocha.tensor([5, 6, 7]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [5, 6, 7]
        assert t2.grad.data.tolist() == -18

    def test_sub_scalar_tensor(self):
        t1 = tocha.tensor(4, requires_grad=True)
        t2 = tocha.tensor([1, 2, 3], requires_grad=True)

        t3 = t1 - t2

        t3.backward(tocha.tensor([5, 6, 7]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == 18
        assert t2.grad.data.tolist() == [-5, -6, -7]

    def test_sub_broadcast(self):
        t1 = tocha.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
        t2 = tocha.tensor([7, 8, 9], requires_grad=True)  # (3,) -> (2,3)

        t3 = t1 - t2

        t3.backward(tocha.tensor([[10, 11, 12], [13, 14, 15]]))

        assert t1.grad is not None
        assert t2.grad is not None
        assert t1.grad.data.tolist() == [[10, 11, 12], [13, 14, 15]]
        assert t2.grad.data.tolist() == [-23, -25, -27]
