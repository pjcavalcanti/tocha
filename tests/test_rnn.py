import unittest
import tocha
import torch
from tocha import nn
from tocha.functional import im2col
import numpy as np

class TestTensorNegate(unittest.TestCase):
    def test_concat_against_torch(self):
        np.random.seed(0)
        for _ in range(100):
            input_size = np.random.randint(1, 10)
            hidden_size = np.random.randint(1, 10)
            num_layers = np.random.randint(1, 10)
            nonlinearity = np.random.choice(["tanh", "relu"]).astype(str)
            bias = np.random.choice([True, False]).astype(bool)
            bias = True
            dropout = 0 # if dropout is not zero, then the output is not the same
            bidirectional = False
            B = np.random.randint(1, 10)
            L = np.random.randint(1, 10)

            x_np = np.random.randn(L, B, input_size).astype(np.float64)
            x = tocha.tensor(x_np, requires_grad=True)
            x_torch = torch.tensor(x_np, requires_grad=True).double()
            
            rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity, bias, dtype="float64")
            rnn_torch = torch.nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=False,
            ).double()
            for name, p in rnn_torch.named_parameters():
                setattr(rnn, name, p.detach().numpy())

            out = rnn(x)
            out_torch = rnn_torch(x_torch)

            assert np.allclose(out[0].data, out_torch[0].detach().numpy(), atol=1e-7)
            # assert np.allclose(out[1].data, out_torch[1].detach().numpy(), atol=1e-7)
            
            grad_np = np.random.randn(*out[0].shape).astype(np.float64)
            grad = tocha.tensor(grad_np, requires_grad=False)
            grad_torch = torch.tensor(grad_np, requires_grad=False)
            # out[0].backward(grad)
            # out_torch[0].backward(grad_torch)
            
            # if not np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-3):
            #     print(f"fail x.grad with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, nonlinearity={nonlinearity}, bias={str(bias)}, batch_size={B}, sequence_length={L}, dropout={dropout}, bidirectional={bidirectional}, B={B}, L={L}")
            # print(f"iteration={_}, \n{x.grad.data=},\n{x_torch.grad.detach().numpy()=}\n\n")
            # if not np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-3):
            #     print(np.allclose(x.grad.data, np.zeros_like(x.grad.data)))
            # assert np.allclose(x.grad.data, x_torch.grad.detach().numpy())#, atol=1e-2), f"fail x.grad with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, nonlinearity={nonlinearity}, bias={str(bias)}, batch_size={B}, sequence_length={L}"
            
