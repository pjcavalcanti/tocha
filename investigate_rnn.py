from typing import Tuple
import torch

# import torch.nn as nn
import tocha
from tocha import Tensor
from tocha.nn import Dropout, Linear
from tocha.module import Parameter, Module
import tocha.functional as F
import numpy as np
from tqdm import tqdm


class RNNCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.bias = bias
        self.hidden_size = hidden_size
        self.activation = F.relu if nonlinearity == "relu" else F.tanh
        self.dtype = np.float32 if dtype == "float32" else np.float64

        self.weight_ih = Parameter(
            np.random.randn(input_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.weight_hh = Parameter(
            np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        if self.bias:
            self.bias_ih = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.bias_hh = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        out = x @ self.weight_ih + h @ self.weight_hh
        if self.bias:
            out += self.bias_ih + self.bias_hh
        return self.activation(out)

class RNN(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: str,
        bias: bool,
        dropout: float = 0.0,
        dtype: str = "float32",
    ) -> None:
        assert (
            input_size > 0
            and hidden_size > 0
            and num_layers > 0
            and nonlinearity in ["relu", "tanh"]
            and dropout >= 0.0
            and dropout <= 1.0
        )
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = Dropout(p=dropout) if dropout > 0.0 else None
        self.activation = F.relu if nonlinearity == "relu" else F.tanh
        self.dtype = np.float32 if dtype == "float32" else np.float64

        for l in range(self.num_layers):
            new_cell = RNNCell(input_size if l==0 else hidden_size, hidden_size, bias, nonlinearity, dtype)                
            self.register_module(f"cell_{l}", new_cell)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        time_length = x.shape[0]
        batch_size = x.shape[1]

        # Initialize hidden states for all time steps and layers
        hidden_states = [
            [
                Tensor(
                    np.zeros((1, batch_size, self.hidden_size), dtype=self.dtype),
                    requires_grad=False,
                )
                for _ in range(self.num_layers)
            ]
            for _ in range(time_length)
        ]
        # Initialize outputs for all time steps
        outputs = []
        for time in range(time_length):
            for c, cell in enumerate(self.children()):
                if c == 0:
                    hidden_states[time][c] = cell(x[time], hidden_states[time - 1][c])
                else:
                    hidden_states[time][c] = cell(
                        hidden_states[time][c - 1], hidden_states[time - 1][c]
                    )
                if self.dropout is not None and self.training and c < self.num_layers - 1:
                    hidden_states[time][c] = self.dropout(hidden_states[time][c])
            # After all layers have been applied, save the output of the last layer
            outputs.append(hidden_states[time][-1])
        outputs = tocha.concatenate(outputs, axis=0)  #
        hidden_states = tocha.concatenate(hidden_states[-1], axis=0)
        return (outputs, hidden_states)

    def apply_layer_i(self, l: int, x_in: Tensor, h_in: Tensor):
        # Apply the weights
        weight_ih = vars(self)[f"weight_ih_l{l}"].transpose((1, 0))
        weight_hh = vars(self)[f"weight_hh_l{l}"].transpose((1, 0))
        h_out = x_in @ weight_ih + h_in @ weight_hh

        # Apply the bias
        if self.bias:
            bias_ih = vars(self)[f"bias_ih_l{l}"]
            bias_hh = vars(self)[f"bias_hh_l{l}"]
            h_out = h_out + bias_ih + bias_hh

        # Apply the activation function
        h_out = self.activation(h_out)

        # Apply dropout
        if l < self.num_layers - 1 and self.dropout is not None and self.training:
            h_out = self.dropout(h_out)
        return h_out

time_length = 3
batch_size = 4
input_size = 2
hidden_size = 5
num_layers = 2

# x = Tensor(np.random.randn(time_length, batch_size, input_size), requires_grad=True)
rnn_torch = RNN(input_size, hidden_size, num_layers, "relu", True)
# z = rnn(x)[0]
# grad = Tensor(np.random.randn(*z.shape))
# z.backward(grad)
# for name, p in rnn.named_parameters():
#     print(name)
#     print(vars(rnn)[cell_name].weight_ih.shape)

np.random.seed(2)
for _ in range(100):
    nonlinearity = str(np.random.choice(["relu", "tanh"]))
    bias = bool(np.random.choice([True, False]))
    dropout = 0
    bidirectional = False
    batch_first = True

    input_size = np.random.randint(1, 6)
    hidden_size = np.random.randint(1, 6)
    num_layers = np.random.randint(1, 6)
    batch_size = np.random.randint(1, 6)
    sequence_length = np.random.randint(1, 6)

    x_np = np.random.randn(sequence_length, batch_size, input_size).astype(np.float32)
    x = tocha.tensor(x_np, requires_grad=True)
    x_torch = torch.tensor(x_np, requires_grad=True)

    rnn_man = RNN(input_size, hidden_size, num_layers, nonlinearity, bias, dropout)
    rnn_torch = torch.nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=nonlinearity,
        bias=bias,
        dropout=dropout,
        bidirectional=bidirectional,
        batch_first=False,
    )
    for name, p in rnn_torch.named_parameters():
        if name.startswith("weight_ih_l"):
            l = int(name[11:])
            vars(rnn_man)[f"cell_{l}"].weight_ih.data = p.detach().numpy().transpose((1, 0))
        if name.startswith("weight_hh_l"):
            l = int(name[11:])
            vars(rnn_man)[f"cell_{l}"].weight_hh.data = p.detach().numpy().transpose((1, 0))
        if name.startswith("bias_ih_l"):
            l = int(name[9:])
            vars(rnn_man)[f"cell_{l}"].bias_ih.data = p.detach().numpy()
        if name.startswith("bias_hh_l"):
            l = int(name[9:])
            vars(rnn_man)[f"cell_{l}"].bias_hh.data = p.detach().numpy()
    

    # for name, p in rnn.named_parameters():
    #     setattr(rnn_man, name, Parameter(p.detach().numpy(), name=name))

    out = rnn_man(x)
    out_torch = rnn_torch(x_torch)

    grad_np = np.random.randn(*out[0].shape).astype(np.float32)  # type: ignore
    grad = tocha.tensor(grad_np, requires_grad=False)
    grad_torch = torch.tensor(grad_np, requires_grad=False)

    out[0].backward(grad)
    out_torch[0].backward(grad_torch)

    passforward = np.allclose(out[0].data, out_torch[0].detach().numpy(), atol=1e-4)
    passgrad = np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-3)  # type: ignore
    assert passforward, f"forward: {passforward}"
    assert passgrad, f"backward: {passgrad}"
    print(passforward and passgrad)
#     debugstr = f"""
# {_}
# backward: {passgrad}, istruezero: {istruezero}:
#             \tinput_size={input_size}
#             \thidden_size={hidden_size}
#             \tnonlinearity={nonlinearity}
#             \tbias={str(bias)}
#             \tnum_layers={num_layers}
#             \tbatch_size={batch_size}
#             \tsequence_length={sequence_length}
# """
