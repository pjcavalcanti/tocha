import torch

# import torch.nn as nn
import tocha
from tocha.module import Parameter
import tocha.nn as nn
import numpy as np
from tqdm import tqdm


# class RNN_man(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias):
#         super().__init__()
#         assert num_layers >= 1, "num_layers must be >= 1"
#         assert nonlinearity in ["relu", "tanh"], "nonlinearity must be 'relu' or 'tanh'"
#         self.bias = bias
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.activation = torch.nn.ReLU() if nonlinearity == "relu" else torch.nn.Tanh()

#         for l in range(num_layers):
#             self.register_parameter(
#                 f"weight_ih_l{l}",
#                 torch.nn.Parameter(
#                     torch.randn(
#                         hidden_size,
#                         input_size if l == 0 else hidden_size,
#                         dtype=torch.float64,
#                     )
#                     / torch.sqrt(torch.tensor(input_size, dtype=torch.float64))
#                 ),
#             )
#             self.register_parameter(
#                 f"weight_hh_l{l}",
#                 torch.nn.Parameter(
#                     torch.randn(hidden_size, hidden_size, dtype=torch.float64)
#                     / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float64))
#                 ),
#             )
#             if bias:
#                 self.register_parameter(
#                     f"bias_ih_l{l}",
#                     torch.nn.Parameter(
#                         torch.randn(hidden_size, dtype=torch.float64)
#                         / torch.sqrt(torch.tensor(input_size, dtype=torch.float64))
#                     ),
#                 )
#                 self.register_parameter(
#                     f"bias_hh_l{l}",
#                     torch.nn.Parameter(
#                         torch.randn(hidden_size, dtype=torch.float64)
#                         / torch.sqrt(torch.tensor(input_size, dtype=torch.float64))
#                     ),
#                 )

#     def forward(self, x):
#         batch_size = x.shape[1]
#         h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float64)
#         params = dict(self.named_parameters())
#         h_out = []
#         for t in range(x.shape[0]):
#             for l in range(0, self.num_layers):
#                 weight_ih = params[f"weight_ih_l{l}"]
#                 weight_hh = params[f"weight_hh_l{l}"]
#                 h[l] = (
#                     x[t] @ weight_ih.t() + h[l] @ weight_hh.t()
#                     if l == 0
#                     else h[l - 1] @ weight_ih.t() + h[l] @ weight_hh.t()
#                 )
#                 if self.bias:
#                     bias_ih = params[f"bias_ih_l{l}"]
#                     bias_hh = params[f"bias_hh_l{l}"]
#                     h[l] += bias_ih + bias_hh
#                 h[l] = self.activation(h[l])
#             h_out.append(h[-1:].clone())
#         return (torch.concatenate(h_out, dim=0), h)


# np.random.seed(1)
# for _ in range(100):
#     nonlinearity = "relu"
#     bias = bool(np.random.choice([True, False]))
#     dropout = 0
#     bidirectional = False
#     batch_first = True

#     input_size = np.random.randint(1, 10)
#     hidden_size = np.random.randint(1, 10)
#     num_layers = np.random.randint(2, 10)
#     B = np.random.randint(1, 10)
#     L = np.random.randint(1, 10)

#     x_np = np.random.randn(L, B, input_size).astype(np.float32)
#     x = tocha.tensor(x_np, requires_grad=True)
#     x_torch = torch.tensor(x_np, requires_grad=True)

#     rnn_man = nn.RNN(input_size, hidden_size, num_layers, nonlinearity, bias, dropout)
#     rnn = torch.nn.RNN(
#         input_size=input_size,
#         hidden_size=hidden_size,
#         num_layers=num_layers,
#         nonlinearity=nonlinearity,
#         bias=bias,
#         dropout=dropout,
#         bidirectional=bidirectional,
#         batch_first=False,
#     )
#     for name, p in rnn.named_parameters():
#         setattr(rnn_man, name, p.detach().numpy())

#     out = rnn_man(x)
#     out_torch = rnn(x_torch)

#     if not np.allclose(out[0].data, out_torch[0].detach().numpy(), atol=1e-7):
#         print(
#             f"fail out[0] with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, nonlinearity={nonlinearity}, bias={str(bias)}, batch_size={B}, sequence_length={L}"
#         )
#     if not np.allclose(out[1].data, out_torch[1].detach().numpy(), atol=1e-7):
#         print(
#             f"fail out[1] with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, nonlinearity={nonlinearity}, bias={str(bias)}, batch_size={B}, sequence_length={L}"
#         )

np.random.seed(2)
for _ in range(10):
    nonlinearity = str(np.random.choice(["relu", "tanh"]))
    bias = bool(np.random.choice([True, False]))
    dropout = 0
    bidirectional = False
    batch_first = True

    input_size = np.random.randint(2, 4)
    hidden_size = np.random.randint(2, 4)
    num_layers = np.random.randint(2, 4)
    B = np.random.randint(1, 4)
    L = np.random.randint(1, 4)

    # print all the parameters for debugging
    # one per line
    # print(f"input_size={input_size}")
    # print(f"hidden_size={hidden_size}")
    # print(f"nonlinearity={nonlinearity}")
    # print(f"bias={str(bias)}")
    # print(f"num_layers={num_layers}")
    # print(f"batch_size={B}")
    # print(f"sequence_length={L}")

    x_np = np.random.randn(L, B, input_size).astype(np.float32)
    x = tocha.tensor(x_np, requires_grad=True)
    x_torch = torch.tensor(x_np, requires_grad=True)

    rnn_man = nn.RNN(input_size, hidden_size, num_layers, nonlinearity, bias, dropout)
    rnn = torch.nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=nonlinearity,
        bias=bias,
        dropout=dropout,
        bidirectional=bidirectional,
        batch_first=False,
    )
    for name, p in rnn.named_parameters():
        setattr(rnn_man, name, Parameter(p.detach().numpy(), name=name))

    out = rnn_man(x)
    out_torch = rnn(x_torch)
    
    # print(np.allclose(out[0].data, out_torch[0].detach().numpy(), atol=1e-7))

    grad_np = np.random.randn(*out[0].shape).astype(np.float64)
    grad = tocha.tensor(grad_np, requires_grad=False)
    grad_torch = torch.tensor(grad_np, requires_grad=False)

    out[0].backward(grad)
    out_torch[0].backward(grad_torch)

    print(
        np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-4),
        np.allclose(x.grad.data, np.zeros_like(x.grad.data), atol=1e-4),
        np.allclose(x_torch.grad.detach().numpy(), np.zeros_like(x.grad.data), atol=1e-4),
    )
    # print(
    #     np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-4)
    #     and np.allclose(x.grad.data, np.zeros_like(x.grad.data), atol=1e-4)
    # )
    break
