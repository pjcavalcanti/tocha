# import unittest
# import torch

# import tocha
# import numpy as np
# import tocha.functional as F
# import tocha.nn as nn



# class TestTensorNegate(unittest.TestCase):
#     def test_batchnorm1d_2dinput(self):
#         np.random.seed(0)
#         for _ in range(10):
#             momentum = np.random.random()
#             n_features = np.random.randint(1, 10)
#             length = np.random.randint(1, 50)
#             batch_size = np.random.randint(1, 16)
#             shape = (batch_size, n_features)

#             a_np = np.random.randn(*shape).astype(np.float64)
#             a = tocha.tensor(a_np, requires_grad = True)
#             a_torch = torch.tensor(a.data, requires_grad = True)


#     norm = nn.BatchNorm1d(n_features, momentum=momentum)
#     norm_torch = torch.nn.BatchNorm1d(n_features,momentum=momentum, dtype=torch.float64)

#     out = norm(a)
#     out_torch = norm_torch(a_torch)


#     print(f"normalizes correctly: {np.allclose(out.data, out_torch.detach().numpy())}")

#     grad_np = np.random.randn(*out.shape)
#     grad = tocha.tensor(grad_np)
#     grad_torch = torch.tensor(grad_np)

#     out.backward(grad)
#     out_torch.backward(grad_torch)

#     assert a.grad is not None
#     assert a_torch.grad is not None

#     print(f"backpropagates correctly: {np.allclose(a.grad.data, a_torch.grad.detach().numpy(), atol = 1e-14)}")

#     norm.eval()
#     norm_torch.eval()

#     a_np = np.random.randn(*shape).astype(np.float64)
#     a = tocha.tensor(a_np, requires_grad = True)
#     a_torch = torch.tensor(a.data, requires_grad = True)


#     out2 = norm(a)
#     out2_torch = norm_torch(a_torch)



#     print(f"eval normalizes correctly: {np.allclose(out2.data, out2_torch.detach().numpy())}")
#     print(f"running mean correct {np.allclose(norm.running_mean.data, norm_torch.running_mean.detach().numpy())}") # type: ignore
#     print(f"running var correct {np.allclose(norm.running_var.data, norm_torch.running_var.detach().numpy())}") # type: ignore
#     print(np.abs((out2.data - out2_torch.detach().numpy())).max())

