import unittest

import itertools
import numpy as np
import torch
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
    def test_multi_dimensional_sum(self):
        np.random.seed(0)  # For reproducibility

        for _ in range(20):  # Number of random tests
            # Random dimensions with varying size
            dims = tuple(np.random.randint(low=1, high=5, size=np.random.randint(1, 6)))

            a_np = np.random.random(tuple(dims)).astype(np.float32)

            # Testing for each axis
            for ax in list(range(len(dims))) + [None] + [()]:
                with self.subTest(f"Testing for dims={dims}, axis={ax}"):
                    a_tocha = tocha.tensor(a_np, requires_grad=True)
                    a_torch = torch.tensor(a_np, requires_grad=True)

                    b_tocha = a_tocha.sum(ax)
                    b_torch = a_torch.sum(ax)

                    self.assertTrue(np.allclose(b_tocha.data, b_torch.detach().numpy()))

                    grad = torch.ones_like(b_torch)
                    b_torch.backward(grad)

                    b_tocha.backward(tocha.tensor(grad.numpy()))
                    self.assertTrue(np.allclose(a_tocha.grad.data, a_torch.grad.numpy())) # type: ignore

            # Testing for all possible tuples of axes
            for ax_tuple in [t for r in range(1, len(dims)+1) for t in itertools.combinations(range(len(dims)), r)]:
                with self.subTest(f"Testing for dims={dims}, axis={ax_tuple}"):
                    a_tocha = tocha.tensor(a_np, requires_grad=True)
                    a_torch = torch.tensor(a_np, requires_grad=True)

                    b_tocha = a_tocha.sum(ax_tuple)
                    b_torch = a_torch.sum(ax_tuple)

                    self.assertTrue(np.allclose(b_tocha.data, b_torch.detach().numpy()))

                    grad = torch.ones_like(b_torch)
                    b_torch.backward(grad)

                    b_tocha.backward(tocha.tensor(grad.numpy()))
                    self.assertTrue(np.allclose(a_tocha.grad.data, a_torch.grad.numpy())) # type: ignore
