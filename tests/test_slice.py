import unittest

import tocha


class TestTensorSlice(unittest.TestCase):
    def test_simple_slice(self):
        t1 = tocha.tensor([1, 2, 3, 4, 5], requires_grad=True)
        t2 = t1[1:4]  # this should result in Tensor([2, 3, 4])

        assert t2.data.tolist() == [2, 3, 4]
        assert t2.requires_grad == True

    def test_backward_propagation(self):
        t1 = tocha.tensor([1, 2, 3, 4, 5], requires_grad=True)
        t2 = t1[1:4]
        t2.backward(tocha.tensor([1, 1, 1]))

        # t1.grad should be Tensor([0, 1, 1, 1, 0]) after backward propagation
        assert t1.grad is not None
        assert t1.grad.data.tolist() == [0, 1, 1, 1, 0]

    def test_backward_2(self):
        t1 = tocha.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], requires_grad=True
        )
        t2 = t1[1, 2:4]

        t2.backward(tocha.tensor([33, 34]))

        assert t2.data.tolist() == [7, 8]
        assert t1.grad.data.tolist() == [  # type: ignore
            [0, 0, 0, 0],
            [0, 0, 33, 34],
            [0, 0, 0, 0],
        ]
