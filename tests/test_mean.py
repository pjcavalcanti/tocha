import unittest

import itertools
import numpy as np
import torch
import tocha


class TestTensorSum(unittest.TestCase):
    def test_multi_dimensional_mean(self):
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

                    b_tocha = a_tocha.mean(ax)
                    b_torch = a_torch.mean(ax)

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

                    b_tocha = a_tocha.mean(ax_tuple)
                    b_torch = a_torch.mean(ax_tuple)

                    self.assertTrue(np.allclose(b_tocha.data, b_torch.detach().numpy()))

                    grad = torch.ones_like(b_torch)
                    b_torch.backward(grad)

                    b_tocha.backward(tocha.tensor(grad.numpy()))
                    self.assertTrue(np.allclose(a_tocha.grad.data, a_torch.grad.numpy())) # type: ignore
