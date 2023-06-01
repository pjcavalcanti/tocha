import unittest
import pytest

import numpy as np
import tocha
import torch


class TestTensorMatMul(unittest.TestCase):
    def test_against_torch(self):
        for _ in range(100):
            nindices1 = int(np.random.randint(1, 5))
            nindices2 = int(np.random.randint(1, 5))
            indices1 = [np.random.randint(1, 10) for _ in range(nindices1)]
            indices2 = [np.random.randint(1, 10) for _ in range(nindices2)]

            if nindices1 == 1 and nindices2 == 1:
                indices1[0] = indices2[0]
            if nindices1 == 1 and nindices2 > 1:
                indices1[0] = indices2[-2]
            if nindices1 > 1 and nindices2 == 1:
                indices1[-1] = indices2[0]
            if nindices1 > 1 and nindices2 > 1:
                indices1[-1] = indices2[-2]
                mindim = min(nindices1, nindices2)
                for i in range(-mindim, -2):
                    indices1[i] = indices2[i]
            indices1 = tuple(indices1)
            indices2 = tuple(indices2)

            xnp = np.random.randn(*indices1).astype(np.float32)  # type: ignore
            ynp = np.random.randn(*indices2).astype(np.float32)  # type: ignore

            x = tocha.tensor(xnp, requires_grad=True)
            y = tocha.tensor(ynp, requires_grad=True)
            z = x @ y

            x_torch = torch.tensor(xnp, requires_grad=True)
            y_torch = torch.tensor(ynp, requires_grad=True)
            z_torch = x_torch @ y_torch

            assert np.allclose(z.data, z_torch.detach().numpy(), atol=1e-5)
            
            gradnp = np.random.randn(*z.shape)
            grad = tocha.tensor(gradnp)
            grad_torch = torch.tensor(gradnp)
            z.backward(grad)
            z_torch.backward(grad_torch)

            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-5) # type: ignore
            assert np.allclose(y.grad.data, y_torch.grad.detach().numpy(), atol=1e-5) # type: ignore
