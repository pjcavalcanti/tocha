import tocha
import tocha.nn as nn
from tocha.functional import relu, sigmoid
import matplotlib.pyplot as plt
import numpy as np

# dummy data
batch_size = 3
in_features = 2
out_features = 3
# image dim
img_height = 5
img_width = 6
x = np.random.randn(batch_size, in_features, img_height, img_width)

# kernel dim
kernel_size = (2, 2)
if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
elif not isinstance(kernel_size, tuple):
    raise TypeError("kernel_size must be int or tuple of int")
kernel = np.random.randn(*kernel_size)
#
convolution = np.zeros(
    (
        img_height - kernel_size[1] + 1,
        img_width - kernel_size[1] + 1,
        img_height,
        img_width,
    )
)  # the convolution kernels
for i in range(img_height - kernel_size[0] + 1):
    for j in range(img_width - kernel_size[1] + 1):
        background = np.zeros((img_height, img_width))
        background[i : i + kernel_size[0], j : j + kernel_size[1]] = kernel
        convolution[i, j] = background

print(convolution.shape)
target = np.random.randint(0, 9, (img_height, img_width))
result = np.tensordot(convolution, target, ((2, 3), (0, 1)))
# print(convolution)
# print(target)
# print(tuple(range(convolution.ndim))[-2:])
# print(tuple(range(convolution.ndim))[-2:], tuple(range(target.ndim))[2:])
# print()
