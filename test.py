import tocha
import numpy as np
import tocha.functional as F

dims = (2, 3, 4)
b = tocha.tensor(np.array([i+1 for i in range(np.prod(dims))]).reshape(dims))
print(b.sum().data)

