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


def test234_2d():
    # THIS WORKS
    # im = (3,4)
    # kernel = (2,2)
    # tensor = (2,) + im.shape = (2,3,4)
    m, n = 3, 4
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([a, b])
    d = im2col2d(c, 2, 2)
    print(c.shape)
    print(c, "\n")
    print(d.shape)
    print(d)


def test2234_2d():
    # THIS WORKS
    # im = (3,4)
    # kernel = (2,2)
    # tensor = (2,2) + im.shape = (2,2,3,4)
    m, n = 3, 4
    a = np.array([i + 1 for i in range(n * m)]).reshape(m, n)
    b = np.array([i + 2 for i in range(n * m)]).reshape(m, n)
    c = np.array([i + 3 for i in range(n * m)]).reshape(m, n)
    d = np.array([i + 4 for i in range(n * m)]).reshape(m, n)
    e = np.array([[a, b], [c, d]])
    f = im2col2d(e, 2, 2)
    print(e.shape)
    print(e, "\n")
    print(f.shape)
    print(f)


test2234_2d()
