import torch
from autograd.tensor import Tensor, Dependency
import numpy as np
from typing import List

def concatenate(ts: List[Tensor], axis: int) -> Tensor:
    data = np.concatenate([t.data for t in ts], axis=axis)
    requires_grad = any([t.requires_grad for t in ts])
    depends_on = []

    if requires_grad:
        for t in ts:
            def grad_fnt(grad: Tensor) -> Tensor:
                shape = t.shape
                
                s = [slice(None)] * t.ndim
                s[axis] = slice(0,shape[axis])
                s = tuple(s)
                new_grad_data = grad.data[s]
                return Tensor(new_grad_data)
            depends_on.append(Dependency(t, grad_fnt))

    return Tensor(data, requires_grad, depends_on)

nindices = np.random.randint(1, 5)
indices = ()
for _ in range(nindices):
    indices = indices + (np.random.randint(1,9),) # type: ignore
axis = np.random.randint(0,nindices)
ntensors = np.random.randint(1,10)
arrays = [np.random.randn(*indices) for n in range(ntensors)]
tensors = [Tensor(arr, requires_grad=True) for arr in arrays]
tensors_torch = [torch.tensor(arr, requires_grad=True) for arr in arrays]
bigt = concatenate(tensors, axis=axis) 
bigt_torch = torch.concatenate(tensors_torch, axis=axis) # type: ignore

grad_np = np.random.randn(*bigt.data.shape)
grad = Tensor(grad_np, requires_grad=False)
grad_torch = torch.tensor(grad_np, requires_grad=False)

bigt.backward(grad)
bigt_torch.backward(grad_torch)

for i in range(len(tensors)):
    print(np.allclose(tensors[i].data, tensors_torch[i].detach().numpy())) # type: ignore
print(np.all(bigt_torch.detach().numpy() == bigt.data))