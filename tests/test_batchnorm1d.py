import unittest
import torch

import tocha
import numpy as np
import tocha.functional as F
import tocha.nn as nn



class TestTensorNegate(unittest.TestCase):
    def test_batchnorm1d_2dinput(self):
        np.random.seed(0)
        
        for _ in range(50):
            batch_size = np.random.randint(2, 32)
            n_features = np.random.randint(1, 10)
            dtype = torch.float64
            device = torch.device("cpu")
            
            momentum = np.random.random()
            eps = np.random.random() * 10 ** (-np.random.randint(1, 10))
            affine=True
            track_running_stats=True

            shape = (batch_size, n_features)
            x = tocha.tensor(np.random.randn(*shape).astype(np.float64),requires_grad=True) # type: ignore
            x_torch = torch.tensor(x.data, requires_grad=True, device=device, dtype=dtype)
            
            norm = nn.BatchNorm1d(n_features, eps=eps, momentum=momentum)
            norm_torch = torch.nn.BatchNorm1d(n_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)
            
            out = norm(x)
            out_torch = norm_torch(x_torch)
            
            grad = tocha.tensor(np.random.randn(*out.shape))
            grad_torch = torch.tensor(grad.data, device=device, dtype=dtype)
            
            out.backward(grad)
            out_torch.backward(grad_torch)
            
            assert np.allclose(out.data, out_torch.detach().numpy())
            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol = 1e-14) # type: ignore
            assert np.allclose(norm.running_mean.data, norm_torch.running_mean.detach().numpy()) # type: ignore
            assert np.allclose(norm.running_var.data, norm_torch.running_var.detach().numpy()) # type: ignore
            
            
            x2 = tocha.tensor(np.random.randn(*shape).astype(np.float64),requires_grad=True)
            x2_torch = torch.tensor(x2.data, requires_grad=True, device=device, dtype=dtype)
            
            norm.eval()
            norm_torch.eval()
            
            out2 = norm(x2)
            out2_torch = norm_torch(x2_torch)
            
            assert np.allclose(out2.data, out2_torch.detach().numpy())
            
            
    def test_batchnorm1d_3dinput(self):
        np.random.seed(0)
        
        for _ in range(50):
            batch_size = np.random.randint(2, 32)
            n_features = np.random.randint(1, 10)
            length = np.random.randint(1, 100)
            dtype = torch.float64
            device = torch.device("cpu")
            
            momentum = np.random.random()
            eps = np.random.random() * 10 ** (-np.random.randint(1, 10))
            affine=True
            track_running_stats=True

            shape = (batch_size, n_features, length)
            x = tocha.tensor(np.random.randn(*shape).astype(np.float64),requires_grad=True)
            x_torch = torch.tensor(x.data, requires_grad=True, device=device, dtype=dtype)
            
            norm = nn.BatchNorm1d(n_features, eps=eps, momentum=momentum)
            norm_torch = torch.nn.BatchNorm1d(n_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)
            
            out = norm(x)
            out_torch = norm_torch(x_torch)
            
            grad = tocha.tensor(np.random.randn(*out.shape))
            grad_torch = torch.tensor(grad.data, device=device, dtype=dtype)
            
            out.backward(grad)
            out_torch.backward(grad_torch)
            
            assert np.allclose(out.data, out_torch.detach().numpy())
            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol = 1e-14) # type: ignore
            assert np.allclose(norm.running_mean.data.squeeze(), norm_torch.running_mean.detach().numpy()) # type: ignore
            assert np.allclose(norm.running_var.data.squeeze(), norm_torch.running_var.detach().numpy()) # type: ignore
            
            
            x2 = tocha.tensor(np.random.randn(*shape).astype(np.float64),requires_grad=True)
            x2_torch = torch.tensor(x2.data, requires_grad=True, device=device, dtype=dtype)
            
            norm.eval()
            norm_torch.eval()
            
            out2 = norm(x2)
            out2_torch = norm_torch(x2_torch)
            
            assert np.allclose(out2.data, out2_torch.detach().numpy())
            