import unittest
import numpy as np
import torch
from autograd.tensor import Tensor
from tocha import nn


class TestTensorNegate(unittest.TestCase):
    def test_conv2dlayer_random_inputs(self):
        for k1 in range(1, 4):
            for k2 in range(1, 4):
                for in_features in range(1, 4):
                    for out_features in range(1, 4):
                        for m in range(k1, 6):
                            for n in range(k2, 6):
                                if not (m >= k1 and n >= k2):
                                    continue
                                # Initialize random seed
                                torch.manual_seed(1)

                                # Generate random 4D tensor
                                x = np.random.rand(2, in_features, m, n).astype(
                                    np.float32
                                )
                                x_torch = torch.from_numpy(x).float()
                                x_manual = Tensor(x)

                                # Generate random weights
                                weight = np.random.rand(
                                    out_features, in_features, k1 * k2
                                ).astype(np.float32)
                                weight_torch = (
                                    torch.from_numpy(weight)
                                    .reshape((out_features, in_features, k1, k2))
                                    .float()
                                )
                                weight_manual = Tensor(weight)

                                # Generate random bias
                                bias = np.random.rand(out_features).astype(np.float32)
                                bias_torch = torch.from_numpy(bias).float()
                                bias_manual = Tensor(bias)

                                # Create layers
                                conv_torch = torch.nn.Conv2d(
                                    in_features, out_features, (k1, k2), bias=True
                                )
                                conv_manual = nn.Conv2d(
                                    in_features, out_features, (k1, k2), bias=True
                                )

                                # Set weights and biases
                                conv_torch.weight = torch.nn.Parameter(weight_torch)
                                conv_torch.bias = torch.nn.Parameter(bias_torch)
                                conv_manual.weight = nn.Parameter(weight_manual)
                                conv_manual.bias = nn.Parameter(bias_manual)

                                # Run both convolutions
                                out_torch = conv_torch(x_torch)
                                out_manual = conv_manual(x_manual)

                                # Check if results are close
                                try:
                                    np.testing.assert_allclose(
                                        out_torch.detach().numpy(),
                                        out_manual.data,
                                        rtol=1e-5,
                                        atol=1e-8,
                                    )
                                except AssertionError:
                                    print(f"out_torch: {out_torch}")
                                    print(f"out_manual: {out_manual}")
                                    print(
                                        f"Shapes do not match for k1={k1}, k2={k2}, in_features={in_features}, out_features={out_features}, m={m}, n={n}"
                                    )
                                    print(f"Shape of input: {x.shape}")
                                    print(f"Shape of weight manual: {weight.shape}")
                                    print(
                                        f"Shape of weight torch: {weight_torch.shape}"
                                    )
                                    print(f"Shape of output (torch): {out_torch.shape}")
                                    print(
                                        f"Shape of output (manual): {out_manual.shape}"
                                    )
                                    raise
