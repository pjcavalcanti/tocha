import unittest
import tocha
import torch
from tocha import nn
from tocha.functional import im2col
import numpy as np


class TestRNN(unittest.TestCase):
    def test_rnn_against_torch_output_0(self):
        np.random.seed(0)
        for _ in range(100):
            input_size = np.random.randint(1, 5)
            hidden_size = np.random.randint(1, 5)
            num_layers = np.random.randint(1, 5)
            nonlinearity = str(np.random.choice(["tanh", "relu"]))
            bias = bool(np.random.choice([True, False]))
            dropout = 0  # if dropout is not zero, then the output is not the same
            bidirectional = False
            B = np.random.randint(1, 5)
            L = np.random.randint(1, 5)

            x_np = np.random.randn(L, B, input_size).astype(np.float64)
            x = tocha.tensor(x_np, requires_grad=True)
            x_torch = torch.tensor(x_np, requires_grad=True).double()

            rnn = nn.RNN(
                input_size, hidden_size, num_layers, nonlinearity, bias, dtype="float64"
            )
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
                if name.startswith("weight_ih_l"):
                    l = int(name[11:])
                    vars(rnn)[f"cell_{l}"].weight_ih.data = p.detach().numpy().transpose((1, 0)).copy()
                if name.startswith("weight_hh_l"):
                    l = int(name[11:])
                    vars(rnn)[f"cell_{l}"].weight_hh.data = p.detach().numpy().transpose((1, 0)).copy()
                if name.startswith("bias_ih_l"):
                    l = int(name[9:])
                    vars(rnn)[f"cell_{l}"].bias_ih.data = p.detach().numpy().copy()
                if name.startswith("bias_hh_l"):
                    l = int(name[9:])
                    vars(rnn)[f"cell_{l}"].bias_hh.data = p.detach().numpy().copy()

            out = rnn(x)
            out_torch = rnn_torch(x_torch)

            assert np.allclose(out[0].data, out_torch[0].detach().numpy(), atol=1e-7)
            assert np.allclose(out[1].data, out_torch[1].detach().numpy(), atol=1e-7)

            grad_np = np.random.randn(*out[0].shape).astype(np.float64) # type: ignore
            grad = tocha.tensor(grad_np, requires_grad=False)
            grad_torch = torch.tensor(grad_np, requires_grad=False)
            out[0].backward(grad)
            out_torch[0].backward(grad_torch)

            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy()) # type: ignore

    def test_rnn_against_torch_output_1(self):
        np.random.seed(0)
        for _ in range(100):
            input_size = np.random.randint(1, 5)
            hidden_size = np.random.randint(1, 5)
            num_layers = np.random.randint(1, 5)
            nonlinearity = str(np.random.choice(["tanh", "relu"]))
            bias = bool(np.random.choice([True, False]))
            dropout = 0  # if dropout is not zero, then the output is not the same
            bidirectional = False
            B = np.random.randint(1, 5)
            L = np.random.randint(1, 5)

            x_np = np.random.randn(L, B, input_size).astype(np.float64)
            x = tocha.tensor(x_np, requires_grad=True)
            x_torch = torch.tensor(x_np, requires_grad=True).double()

            rnn = nn.RNN(
                input_size, hidden_size, num_layers, nonlinearity, bias, dtype="float64"
            )
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
                if name.startswith("weight_ih_l"):
                    l = int(name[11:])
                    vars(rnn)[f"cell_{l}"].weight_ih.data = p.detach().numpy().transpose((1, 0)).copy()
                if name.startswith("weight_hh_l"):
                    l = int(name[11:])
                    vars(rnn)[f"cell_{l}"].weight_hh.data = p.detach().numpy().transpose((1, 0)).copy()
                if name.startswith("bias_ih_l"):
                    l = int(name[9:])
                    vars(rnn)[f"cell_{l}"].bias_ih.data = p.detach().numpy().copy()
                if name.startswith("bias_hh_l"):
                    l = int(name[9:])
                    vars(rnn)[f"cell_{l}"].bias_hh.data = p.detach().numpy().copy()

            out = rnn(x)
            out_torch = rnn_torch(x_torch)

            assert np.allclose(out[0].data, out_torch[0].detach().numpy(), atol=1e-7)
            assert np.allclose(out[1].data, out_torch[1].detach().numpy(), atol=1e-7)

            grad_np = np.random.randn(*out[1].shape).astype(np.float64) # type: ignore
            grad = tocha.tensor(grad_np, requires_grad=False)
            grad_torch = torch.tensor(grad_np, requires_grad=False)
            out[1].backward(grad)
            out_torch[1].backward(grad_torch)

            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy()) # type: ignore
