import tocha
import tocha.nn as nn
from tocha.functional import relu, sigmoid
import matplotlib.pyplot as plt
import numpy as np


# kernel dim
kernel_size = (2, 2)
kernel_height = kernel_size[0]
kernel_width = kernel_size[1]
kernel = np.ones(kernel_size)
# image dim
img_height = 5
img_width = 6
#
convolution = np.zeros(
    (img_height - kernel_width + 1, img_width - kernel_width + 1, img_height, img_width)
)  # the convolution kernels
for i in range(img_height - kernel_height + 1):
    for j in range(img_width - kernel_width + 1):
        background = np.zeros((img_height, img_width))
        background[i : i + kernel_height, j : j + kernel_width] = kernel
        convolution[i, j] = background

target = np.random.randint(0, 9, (img_height, img_width))
result = np.tensordot(convolution, target, ((2, 3), (0, 1)))
print(convolution)
print(target)
print(tuple(range(convolution.ndim))[-2:])
print(tuple(range(convolution.ndim))[-2:], tuple(range(target.ndim))[2:])
print()
