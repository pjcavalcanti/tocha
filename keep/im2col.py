import torch
import numpy as np


def im2col2d(im, k1, k2):
    m, n = im.shape[-2:]
    cols = im[..., 0:k1, 0:k2]
    cols = cols.reshape(im.shape[:-2] + (k1 * k2, 1))
    for i in range(0, m - k1 + 1):
        for j in range(0, n - k2 + 1):
            if (i, j) == (0, 0):
                continue
            ncol = im[..., i : i + k1, j : j + k2]
            ncol = ncol.reshape(im.shape[:-2] + (k1 * k2, 1))
            cols = np.concatenate([cols, ncol], axis=-1)
    return cols


def im2row2d(im, k1, k2):
    m, n = im.shape[-2:]
    cols = im[..., 0:k1, 0:k2]
    cols = cols.reshape(im.shape[:-2] + (1, k1 * k2))
    for i in range(0, m - k1 + 1):
        for j in range(0, n - k2 + 1):
            if (i, j) == (0, 0):
                continue
            ncol = im[..., i : i + k1, j : j + k2]
            ncol = ncol.reshape(im.shape[:-2] + (1, k1 * k2))
            cols = np.concatenate([cols, ncol], axis=-2)
    return cols


def test234_2d():
    # THIS WORKS
    # im = (3,4)
    # kernel = (2,2)
    # tensor = (2,) + im.shape = (2,3,4)
    m, n = 3, 4
    k1, k2 = 2, 2
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([a, b])
    d = im2col2d(c, k1, k2)
    e = im2row2d(c, k1, k2)
    print(c.shape)
    print(c, "\n")
    print(d.shape)
    print(d)
    print("\ntranspose:\n")
    print(e)


def test2234_2d():
    # THIS WORKS
    # im = (3,4)
    # kernel = (2,2)
    # tensor = (2,2) + im.shape = (2,2,3,4)
    m, n = 3, 4
    k1, k2 = 2, 2
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([i + 3 for i in range(n * m)]).reshape(m, n)
    d = np.array([i + 4 for i in range(n * m)]).reshape(m, n)
    e = np.array([[a, b], [c, d]])
    f = im2col2d(e, k1, k2)
    g = im2row2d(e, k1, k2)
    print(e.shape)
    print(e, "\n")
    print(f.shape)
    print(f)
    print("\ntranspose:\n")
    print(g)


def test_kern1():
    m, n = 3, 4
    k1, k2 = 2, 2
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([i + 3 for i in range(n * m)]).reshape(m, n)
    d = np.array([i + 4 for i in range(n * m)]).reshape(m, n)
    e = np.array([[a, b], [c, d]])  # (2,2,3,4)
    f = im2row2d(e, k1, k2)  # (2, 2, 6, 4), there are 6 2x2 submatrices of size 2x2

    kern1 = np.array([1 for i in range(k1 * k2)]).reshape(k1 * k2)  # (4,)
    kern = kern1
    print(f"kern: {kern.shape}")
    g = np.tensordot(f, kern, axes=((-1), (0,)))  # (2,2,6,2)
    h = np.reshape(g, (2, 2, 2, 3))  # (2,2,2,6)
    print(f"input.shape: {e.shape}")
    print(f"output.shape: {h.shape}")
    print(f"input:\n{e}")
    print(f"output:\n{h}")
    print("This is summing the numbers at each 2x2 submatrix of the 3x4 matrix")


def test_kern2():
    m, n = 3, 4
    k1, k2 = 2, 2
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([i + 3 for i in range(n * m)]).reshape(m, n)
    d = np.array([i + 4 for i in range(n * m)]).reshape(m, n)
    e = np.array([[a, b], [c, d]])  # (2,2,3,4)
    f = im2row2d(e, k1, k2)  # (2, 2, 6, 4), there are 6 2x2 submatrices of size 2x2

    kern1 = np.array([1 for i in range(k1 * k2)]).reshape(k1 * k2)  # (4,)
    kern2 = np.array([2 for i in range(k1 * k2)]).reshape(k1 * k2)  # (4,)
    kern = np.array([[kern1, kern2]])  # (c_out, c_in, k1*k2)
    g = np.tensordot(f, kern, axes=((-3, -1), (1, 2)))  # (2,2,6,2) (2,4)
    # print(f"tensordot: {g.shape}")
    b_out = e.shape[0]
    c_out = kern.shape[0]
    x_out = e.shape[-2] - k1 + 1
    y_out = e.shape[-1] - k2 + 1
    h = np.reshape(g, (b_out, c_out, x_out, y_out))  # (2,2,2,6)

    print(f"kernel:\n{kern.reshape(2, k1, k2)} = (C_out, C_in, k1, k2)")
    print(f"input.shape: {e.shape}  = (B, C_in, W, H)")
    print(f"output.shape: {h.shape} = (B, C_out, W-k1+1, H-k2+1)")
    print(f"input:\n{e}")
    print(f"output:\n{h}")
    print(f"probably correct")


def conv2drow(x, in_features, out_features, kernel_size):
    k1, k2 = kernel_size
    kern1 = np.array([1 for i in range(k1 * k2)]).reshape(k1 * k2)  # (4,)
    kern2 = np.array([2 for i in range(k1 * k2)]).reshape(k1 * k2)  # (4,)
    kernel = np.array(
        [[[i + 1 for _ in range(k1 * k2)] for i in range(in_features)]]
    )  # (c_out, c_in, k1*k2)
    bias = np.array([3 for _ in range(out_features)])
    print(kernel.shape)
    out = im2row2d(x, k1, k2)
    out = np.tensordot(out, kernel, axes=((-3, -1), (1, 2)))  # (2,2,6,2) (2,4)

    b_out = x.shape[0]
    c_out = kernel.shape[0]
    x_out = x.shape[-2] - k1 + 1
    y_out = x.shape[-1] - k2 + 1
    out = np.reshape(out, (b_out, c_out, x_out, y_out)) + bias  # (2,2,2,6)

    return out


def conv2dcol(x, in_features, out_features, kernel_size):
    k1, k2 = kernel_size
    kern1 = np.array([1 for i in range(k1 * k2)]).reshape(k1 * k2)  # (4,)
    kern2 = np.array([2 for i in range(k1 * k2)]).reshape(k1 * k2)  # (4,)
    kernel = np.array(
        [
            [[i + 1 for _ in range(k1 * k2)] for i in range(in_features)]
            for _ in range(out_features)
        ]
    )  # (c_out, c_in, k1*k2)
    bias = np.array([3 for _ in range(out_features)])
    print(kernel.shape)
    out = im2col2d(x, k1, k2)
    out = np.tensordot(out, kernel, axes=((-3, -2), (1, 2)))  # (2,2,6,2) (2,4)

    b_out = x.shape[0]
    c_out = kernel.shape[0]
    x_out = x.shape[-2] - k1 + 1
    y_out = x.shape[-1] - k2 + 1
    out = np.reshape(out, (b_out, c_out, x_out, y_out)) + bias  # (2,2,2,6)

    return out


def test_conv2d():
    m, n = 3, 4
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([i + 3 for i in range(n * m)]).reshape(m, n)
    d = np.array([i + 4 for i in range(n * m)]).reshape(m, n)
    x = np.array([[a, b], [c, d]])  # (2,2,3,4)

    print(x.shape)
    out = conv2drow(x, 2, 1, (2, 2))
    out2 = conv2dcol(x, 2, 2, (2, 2))
    print(out)
    print("\n\n")
    print(out2)


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

    print(x.shape)
    print(conv.weight.shape)
    print(vars(conv))
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
    out2 = conv2drow(x, in_features, out_features, (k1, k2))

    print("Checking if torch conv2d equals pytorch implementation")
    print(out.int().numpy() == out2)


# test_conv2d()
test_conv2dtorch()
