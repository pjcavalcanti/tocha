import tocha
import numpy as np
import tocha.functional as F

b, c, h, w = 1, 1, 3, 3
a = np.array([i + 1 for i in range(b * c * h * w)]).reshape(b, c, h, w)
a = tocha.tensor(a, requires_grad=True)

t = F.im2col(a, (2, 2))
print(a.data)
print(t.data)
t.backward(tocha.tensor(np.ones(t.shape)))
print(a.grad.data)  # type: ignore
