import torch
# import torch.nn as nn
import tocha.nn as nn


class RNN_man(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert nonlinearity in ["relu", "tanh"], "nonlinearity must be 'relu' or 'tanh'"
        self.bias = bias
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = torch.nn.ReLU() if nonlinearity == "relu" else torch.nn.Tanh()

        for l in range(num_layers):
            self.register_parameter(
                f"weight_ih_l{l}",
                torch.nn.Parameter(
                    torch.randn(
                        hidden_size,
                        input_size if l == 0 else hidden_size,
                        dtype=torch.float64,
                    )
                    / torch.sqrt(torch.tensor(input_size, dtype=torch.float64))
                ),
            )
            self.register_parameter(
                f"weight_hh_l{l}",
                torch.nn.Parameter(
                    torch.randn(hidden_size, hidden_size, dtype=torch.float64)
                    / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float64))
                ),
            )
            if bias:
                self.register_parameter(
                    f"bias_ih_l{l}",
                    torch.nn.Parameter(
                        torch.randn(hidden_size, dtype=torch.float64)
                        / torch.sqrt(torch.tensor(input_size, dtype=torch.float64))
                    ),
                )
                self.register_parameter(
                    f"bias_hh_l{l}",
                    torch.nn.Parameter(
                        torch.randn(hidden_size, dtype=torch.float64)
                        / torch.sqrt(torch.tensor(input_size, dtype=torch.float64))
                    ),
                )

    def forward(self, x):
        batch_size = x.shape[1]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float64)
        params = dict(self.named_parameters())
        h_out = []
        for t in range(x.shape[0]):
            for l in range(0, self.num_layers):
                weight_ih = params[f"weight_ih_l{l}"]
                weight_hh = params[f"weight_hh_l{l}"]
                h[l] = (
                    x[t] @ weight_ih.t() + h[l] @ weight_hh.t()
                    if l == 0
                    else h[l - 1] @ weight_ih.t() + h[l] @ weight_hh.t()
                )
                if self.bias:
                    bias_ih = params[f"bias_ih_l{l}"]
                    bias_hh = params[f"bias_hh_l{l}"]
                    h[l] += bias_ih + bias_hh
                h[l] = self.activation(h[l])
            h_out.append(h[-1:].clone())
        return (torch.concatenate(h_out, dim=0), h)


nonlinearity = "relu"
bias = True
dropout = 0
bidirectional = False
batch_first = True

torch.manual_seed(3)
input_size = int(torch.randint(1, 10, size=()))
hidden_size = int(torch.randint(1, 10, size=()))
num_layers = int(torch.randint(1, 10, size=()))
B = int(torch.randint(1, 10, size=()))
L = int(torch.randint(1, 10, size=()))

rnn_man = nn.RNN(input_size, hidden_size, num_layers, nonlinearity, bias, dropout)


# for _ in range(100):
#     input_size = int(torch.randint(1, 10, size=()))
#     hidden_size = int(torch.randint(1, 10, size=()))
#     num_layers = int(torch.randint(1, 10, size=()))
#     B = int(torch.randint(1, 10, size=()))
#     L = int(torch.randint(1, 10, size=()))

#     x = torch.randn(L, B, (input_size)).double()
#     rnn_man = RNN_man(input_size, hidden_size, num_layers, nonlinearity, bias).double()
#     rnn = torch.nn.RNN(
#         input_size=input_size,
#         hidden_size=hidden_size,
#         num_layers=num_layers,
#         nonlinearity=nonlinearity,
#         bias=bias,
#         dropout=dropout,
#         bidirectional=bidirectional,
#         batch_first=False,
#     ).double()
#     for name, p in rnn.named_parameters():
#         setattr(rnn_man, name, p)

#     out_man = rnn_man(x)
#     out = rnn(x)

#     if not torch.allclose(out[0], out_man[0], atol=1e-18):
#         print(
#             f"fail out[0] with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, nonlinearity={nonlinearity}, bias={str(bias)}, batch_size={B}, sequence_length={L}"
#         )
#     if not torch.allclose(out[1], out_man[1], atol=1e-19):
#         print(
#             f"fail out[1] with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, nonlinearity={nonlinearity}, bias={str(bias)}, batch_size={B}, sequence_length={L}"
#         )
