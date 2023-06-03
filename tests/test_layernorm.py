import unittest
import numpy as np
import tocha
import tocha.nn
import torch

# write a test for the LayerNorm module
class LayerNormTest(unittest.TestCase):
    def test_layerNorm_against_torch(self):
        np.random.seed(0)
        for _ in range(100):
            # prepare data
            ndims = np.random.randint(2, 10)
            shape = np.random.randint(1, 5, size=ndims)
            eps = np.random.rand() * 10**(np.random.randint(-10, -1))
            elementwise_affine = bool(np.random.choice([True, False]))
            
            xnp = np.random.randn(*shape).astype(np.float64) # type: ignore
            normalized_shape = list(xnp.shape[1:])
            x_tocha = tocha.tensor(xnp, requires_grad=True)
            x_torch = torch.tensor(xnp, requires_grad=True)

            # initialize and equate layers
            norm_torch = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, dtype=torch.float64)
            norm_tocha = tocha.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
            if elementwise_affine:
                norm_tocha.weight.data = norm_torch.weight.detach().numpy()
                norm_tocha.bias.data = norm_torch.bias.detach().numpy()
            out_tocha = norm_tocha(x_tocha)
            out_torch = norm_torch(x_torch)

            # check forward and backward pass
            passforward = np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-10)
            assert passforward, 'forward pass failed'

            gradnp = np.random.randn(*out_tocha.shape).astype(np.float64) # type: ignore
            grad_tocha = tocha.tensor(gradnp)
            grad_torch = torch.tensor(gradnp)

            out_tocha.backward(grad_tocha)
            out_torch.backward(grad_torch)

            passbackward = np.allclose(x_tocha.grad.data, x_torch.grad.detach().numpy(), atol=1e-10) # type: ignore
            assert passbackward, 'backward pass failed'