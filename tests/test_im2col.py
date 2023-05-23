import unittest
import tocha
from tocha import nn
from tocha.functional import im2col_slow


class TestTensorNegate(unittest.TestCase):
    def test_conv2dlayer_random_inputs_shape(self):
        maxn = 5
        for k1 in range(1, maxn):
            for k2 in range(1, maxn):
                for B in range(1, maxn):
                    for H in range(k1, k1 + maxn):
                        for W in range(k2, k2 + maxn):
                            for Cin in range(1, maxn):
                                dims = [B, Cin, H, W]
                                product = 1
                                for dim in dims:
                                    product *= dim
                                x = tocha.tensor([i + 1 for i in range(product)])
                                x = x.reshape(tuple(dims))

                                kernel_size = (k1, k2)
                                out = im2col_slow(x, kernel_size)
                                assert out.shape == (
                                    B,
                                    Cin,
                                    k1 * k2,
                                    (H - k1 + 1) * (W - k2 + 1),
                                )
