import unittest
import tocha
import torch
from tocha import nn
from tocha.functional import im2col
import numpy as np


class TestTensorNegate(unittest.TestCase):
    def test_concat_against_torch(self):
        np.random.seed(0)
        for _ in range(15):
            # random arrays
            nindices = np.random.randint(1, 5)
            indices = ()
            for _ in range(nindices):
                indices = indices + (np.random.randint(1, 9),)
            ntensors = np.random.randint(1, 10)
            arrays = [np.random.randn(*indices) for n in range(ntensors)]
            # random concat axis
            axis = np.random.randint(0, nindices)

            # do everything in tocha
            tensors = [tocha.tensor(arr, requires_grad=True) for arr in arrays]
            bigt = tocha.concatenate(tensors, axis=axis)
            grad_np = np.random.randn(*bigt.data.shape)
            grad = tocha.tensor(grad_np, requires_grad=False)
            bigt.backward(grad)

            # do the same in torch
            tensors_torch = [torch.tensor(arr, requires_grad=True) for arr in arrays]
            bigt_torch = torch.concatenate(tensors_torch, axis=axis)  # type: ignore
            grad_torch = torch.tensor(grad_np, requires_grad=False)
            bigt_torch.backward(grad_torch)

            # check for agreement
            assert np.all(bigt_torch.detach().numpy() == bigt.data)
            for i in range(len(tensors)):
                assert np.allclose(
                    tensors[i].grad.data,
                    tensors_torch[i].grad.detach().numpy(),
                    atol=1e-6,
                )
