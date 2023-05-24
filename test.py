import tocha
import torch
import numpy as np
import tocha.functional as F
import tocha.nn as nn


Bnp = np.random.randn(5, 4, 3, 2)
Anp = np.random.randn(2, 3, 4, 5)

A = tocha.tensor(Anp, requires_grad=True)
B = tocha.tensor(Bnp, requires_grad=True)
A_torch = torch.tensor(Anp, requires_grad=True, dtype=torch.float32)
B_torch = torch.tensor(Bnp, requires_grad=True, dtype=torch.float32)

C = tocha.tensordot(A, B, axes=((1,2),(2,1)))
C_torch = torch.tensordot(A_torch, B_torch, dims=((1,2),(2,1))) # type: ignore
grad = tocha.tensor(np.ones_like(C).astype(np.float32))
grad_torch = torch.ones_like(C_torch)
C.backward(grad)
C_torch.backward(grad_torch)

print(np.allclose(A.grad.data, A_torch.grad.numpy())) # type: ignore