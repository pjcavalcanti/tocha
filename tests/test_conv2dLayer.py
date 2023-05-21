import unittest
import numpy as np
import torch
import tocha
from tocha import nn


class TestTensorNegate(unittest.TestCase):
    def test_conv2layer_shapes(self):
        maxn = 4
        for k1 in range(1, maxn):
            for k2 in range(1, maxn):
                for B in range(1, maxn):
                    for H in range(k1, k1 + maxn):
                        for W in range(k2, k2 + maxn):
                            for Cin in range(1, maxn):
                                for Cout in range(1, maxn):
                                    dims = [B, Cin, H, W]
                                    product = 1
                                    for dim in dims:
                                        product *= dim
                                    x = tocha.tensor([i + 1 for i in range(product)])
                                    x = x.reshape(tuple(dims))

                                    kernel_size = (k1, k2)
                                    conv = nn.Conv2d(Cin, Cout, kernel_size, bias=False)
                                    out = conv(x)
                                    assert out.shape == (
                                        B,
                                        Cout,
                                        H - k1 + 1,
                                        W - k2 + 1,
                                    )

    def test_conv2dlayer_random_inputs1(self):
        maxn = 4
        for k1 in range(1, maxn):
            for k2 in range(1, maxn):
                for B in range(1, maxn):
                    for H in range(k1, k1 + maxn):
                        for W in range(k2, k2 + maxn):
                            for Cin in range(1, maxn):
                                for Cout in range(1, maxn):
                                    dims = [B, Cin, H, W]
                                    x_np = np.random.randn(*dims)
                                    x = tocha.tensor(x_np)
                                    x_torch = torch.from_numpy(x_np)

                                    kernel_size = (k1, k2)
                                    conv = nn.Conv2d(Cin, Cout, kernel_size, bias=True)
                                    conv_torch = torch.nn.Conv2d(
                                        Cin, Cout, kernel_size, bias=True
                                    )
                                    bias_np = np.random.randn(Cout)
                                    bias = tocha.tensor(bias_np)
                                    bias_torch = torch.from_numpy(bias_np)
                                    weight_np = np.random.randn(Cout, Cin, k1, k2)
                                    weight = tocha.tensor(
                                        weight_np.reshape(Cout, Cin, k1 * k2)
                                    )
                                    weight_torch = torch.from_numpy(weight_np)

                                    conv.weight = nn.Parameter(weight)
                                    conv.bias = nn.Parameter(bias)
                                    conv_torch.weight = torch.nn.Parameter(weight_torch)
                                    conv_torch.bias = torch.nn.Parameter(bias_torch)

                                    out = conv(x)
                                    out_torch = conv_torch(x_torch)
                                    assert out.shape == out_torch.shape
