import torch
import tocha
import tocha.nn as nn
from tocha.functional import relu, sigmoid, flatten, tanh, im2col2d, im2row2d
import matplotlib.pyplot as plt
import numpy as np
import autograd.tensor as at
from autograd.tensor import Tensor, tensordot


def conv2dcol(x, in_features, out_features, kernel_size):
    k1, k2 = kernel_size
    kernel = tocha.tensor(
        [
            [[i + 1 for _ in range(k1 * k2)] for i in range(in_features)]
            for _ in range(out_features)
        ]
    )  # (c_out, c_in, k1*k2)
    bias = tocha.tensor([3 for _ in range(out_features)])
    print(kernel.shape)
    out = im2col2d(x, (k1, k2))
    out = tensordot(out, kernel, axes=((-3, -2), (1, 2)))  # (2,2,6,2) (2,4)

    b_out = x.shape[0]
    c_out = kernel.shape[0]
    x_out = x.shape[-2] - k1 + 1
    y_out = x.shape[-1] - k2 + 1
    out = out.reshape((b_out, c_out, x_out, y_out)) + bias  # (2,2,2,6)

    return out


def test_conv2dtorch():
    torch.manual_seed(1)
    in_features, out_features = 2, 1
    k1, k2 = 2, 2
    conv = torch.nn.Conv2d(in_features, out_features, (k1, k2), bias=True)
    m, n = 3, 4
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([i + 3 for i in range(n * m)]).reshape(m, n)
    d = np.array([i + 4 for i in range(n * m)]).reshape(m, n)
    x = np.array([[a, b], [c, d]])  # (2,2,3,4)
    x = torch.from_numpy(x).float()

    weight = (
        torch.tensor(
            [[[i + 1 for _ in range(2 * 2)] for i in range(2)] for _ in range(1)]
        )
        .reshape((out_features, in_features, k1, k2))
        .float()
    )
    bias = torch.tensor([3]).float()
    conv.weight = torch.nn.Parameter(weight)
    conv.bias = torch.nn.Parameter(bias)

    out = conv(x)
    out2 = conv2dcol(x, in_features, out_features, (k1, k2))

    print("Checking if torch conv2d equals pytorch implementation")
    print(out.int().numpy() == out2.data)


def testconv2dlayertorch():
    torch.manual_seed(1)
    in_features, out_features = 2, 1
    k1, k2 = 2, 2
    conv = torch.nn.Conv2d(in_features, out_features, (k1, k2), bias=True)
    convmanual = nn.Conv2d(in_features, out_features, (k1, k2), bias=True)
    m, n = 3, 4
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([i + 3 for i in range(n * m)]).reshape(m, n)
    d = np.array([i + 4 for i in range(n * m)]).reshape(m, n)
    x = np.array([[a, b], [c, d]])  # (2,2,3,4)
    x = torch.from_numpy(x).float()

    weight = (
        torch.tensor(
            [[[i + 1 for _ in range(2 * 2)] for i in range(2)] for _ in range(1)]
        )
        .reshape((out_features, in_features, k1, k2))
        .float()
    )

    bias = torch.tensor([3]).float()
    conv.weight = torch.nn.Parameter(weight)
    conv.bias = torch.nn.Parameter(bias)
    convmanual.weight = nn.Parameter(
        weight.numpy().reshape((out_features, in_features, k1 * k2))
    )
    convmanual.bias = nn.Parameter(bias.numpy())

    out = conv(x)
    out2 = convmanual(x)

    print(out, out2)

    print("Checking if torch conv2d equals pytorch implementation")
    print(out.int().numpy() == out2.data)


testconv2dlayertorch()
