import unittest
import pytest

import numpy as np
import tocha
import torch


class TestTensorMatMul(unittest.TestCase):
    def test_simple_matmul(self):
        # t1 is (3, 2)
        t1 = tocha.tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)

        # t2 is a (2, 1)
        t2 = tocha.tensor([[10], [20]], requires_grad=True)

        t3 = t1 @ t2

        assert t3.data.tolist() == [[50], [110], [170]]

        grad = tocha.tensor([[-1], [-2], [-3]])
        t3.backward(grad)

        np.testing.assert_array_equal(t1.grad.data, grad.data @ t2.data.T)  # type: ignore

        np.testing.assert_array_equal(t2.grad.data, t1.data.T @ grad.data)  # type: ignore
    def test_against_torch(self):
        for _ in range(100):
            nindices1 = int(np.random.randint(1,5))
            nindices2 = int(np.random.randint(1,5))
            indices1 = []
            indices2 = []
            for _ in range(nindices1):
                indices1.append(np.random.randint(1,10))
            for _ in range(nindices2):
                indices2.append(np.random.randint(1,10))
            indices2[0] = indices1[-1] 
            indices1 = tuple(indices1)
            indices2 = tuple(indices2)
            
            xnp = np.random.randn(*indices1).astype(np.float32) # type: ignore
            ynp = np.random.randn(*indices2).astype(np.float32) # type: ignore
            print(xnp.dtype, ynp.dtype)
            
            x = tocha.tensor(xnp, requires_grad=True)
            y = tocha.tensor(ynp, requires_grad=True)
            z = x @ y
            
            x_torch = torch.tensor(xnp, requires_grad=True)
            y_torch = torch.tensor(ynp, requires_grad=True)
            z_torch = x_torch @ y_torch
            
            print(x.shape, y.shape, z.shape)
            print(x_torch.shape, y_torch.shape, z_torch.shape)
            assert np.allclose(z.data, z_torch.detach().numpy(), atol=1e-5)
        
            
        