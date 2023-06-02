import torch
import tocha
from tocha.module import Module, Parameter
from tocha.nn import LSTMCell
import tocha.functional as F
from tocha import Tensor
from typing import Tuple
import numpy as np

from tocha.nn import Dropout


def equate_cell_params(
    lstm_cell_man: tocha.nn.LSTMCell,
    lstm_torch: torch.nn.LSTMCell,
    hidden_size: int,
    bias,
):
    weight_ih = lstm_torch.weight_ih
    weight_hh = lstm_torch.weight_hh
    bias_ih = lstm_torch.bias_ih
    bias_hh = lstm_torch.bias_hh

    lstm_cell_man.i_weight_ih.data = weight_ih[:hidden_size, :].t().detach().numpy()
    lstm_cell_man.i_weight_hh.data = weight_hh[:hidden_size, :].t().detach().numpy()
    lstm_cell_man.f_weight_ih.data = (
        weight_ih[hidden_size : 2 * hidden_size, :].t().detach().numpy()
    )
    lstm_cell_man.f_weight_hh.data = (
        weight_hh[hidden_size : 2 * hidden_size, :].t().detach().numpy()
    )
    lstm_cell_man.g_weight_ih.data = (
        weight_ih[2 * hidden_size : 3 * hidden_size, :].t().detach().numpy()
    )
    lstm_cell_man.g_weight_hh.data = (
        weight_hh[2 * hidden_size : 3 * hidden_size, :].t().detach().numpy()
    )
    lstm_cell_man.o_weight_ih.data = (
        weight_ih[3 * hidden_size :, :].t().detach().numpy()
    )
    lstm_cell_man.o_weight_hh.data = (
        weight_hh[3 * hidden_size :, :].t().detach().numpy()
    )
    if bias:
        lstm_cell_man.i_bias_ih.data = bias_ih[:hidden_size].t().detach().numpy()
        lstm_cell_man.i_bias_hh.data = bias_hh[:hidden_size].t().detach().numpy()
        lstm_cell_man.f_bias_ih.data = (
            bias_ih[hidden_size : 2 * hidden_size].t().detach().numpy()
        )
        lstm_cell_man.f_bias_hh.data = (
            bias_hh[hidden_size : 2 * hidden_size].t().detach().numpy()
        )
        lstm_cell_man.g_bias_ih.data = (
            bias_ih[2 * hidden_size : 3 * hidden_size].t().detach().numpy()
        )
        lstm_cell_man.g_bias_hh.data = (
            bias_hh[2 * hidden_size : 3 * hidden_size].t().detach().numpy()
        )
        lstm_cell_man.o_bias_ih.data = bias_ih[3 * hidden_size :].t().detach().numpy()
        lstm_cell_man.o_bias_hh.data = bias_hh[3 * hidden_size :].t().detach().numpy()




# for n, p in lstm_torch.named_parameters():
#     print(n, p.shape)

# cell0 = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
# cell1 = torch.nn.LSTMCell(hidden_size, hidden_size, bias=bias)

# cells = [cell0, cell1]

# xnp = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
# x_torch = torch.tensor(xnp, requires_grad=True)

# cell0.weight_ih = lstm_torch.weight_ih_l0
# cell0.weight_hh = lstm_torch.weight_hh_l0
# cell1.weight_ih = lstm_torch.weight_ih_l1
# cell1.weight_hh = lstm_torch.weight_hh_l1

# hs = [torch.zeros(batch_size, hidden_size) for _ in range(num_layers)]
# cs = [torch.zeros(batch_size, hidden_size) for _ in range(num_layers)]
# outs = []
# for i in range(seq_len):
#     h = x_torch[i]
#     for j in range(num_layers):
#         h, c = cells[j](h, (hs[j], cs[j]))
#         hs[j] = h
#         cs[j] = c
#     outs.append(h.unsqueeze(0))
# outs = torch.concatenate(outs, dim=0)
# state = (torch.stack(hs, dim=0), torch.stack(cs, dim=0))
# out_torch, (h_torch, c_torch) = lstm_torch(x_torch)

# print(h_torch.shape, c_torch[1].shape)
# print(state[0].shape, state[1].shape)
# print(torch.allclose(outs, out_torch))

# for i in range(num_layers):
#     print(torch.allclose(h_torch[i], state[0][i]))
#     print(torch.allclose(c_torch[i], state[1][i]))


class LSTM(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = Dropout(p=dropout) if dropout > 0 else None
        
        for i in range(num_layers):
            new_cell = LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            self.register_module(f"cell{i}", new_cell)
            
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        
        hs = [Tensor(np.zeros((1, batch_size, hidden_size))) for _ in range(num_layers)]
        cs = [Tensor(np.zeros((1, batch_size, hidden_size))) for _ in range(num_layers)]
        
        x_outs = []
        for t in range(seq_len):
            x_in = x[t]
            for l, cell in enumerate(self.children()):
                h, c = cell(x_in, (hs[l], cs[l]))
                hs[l] = h
                cs[l] = c
                if self.dropout is not None and l < self.num_layers - 1:
                    h = self.dropout(h)
                x_in = h
            x_outs.append(h)
        x_outs = tocha.concatenate(x_outs, axis=0)
        hs = tocha.concatenate(hs, axis=0)
        cs = tocha.concatenate(cs, axis=0)
        return x_outs, (hs, cs)
    
    def children(self):
        for child in super().children():
            if isinstance(child, LSTMCell):
                yield child
            else:
                continue
            
print("\nSTART\n")

np.random.seed(2)

seq_len = int(np.random.randint(1, 4))
batch_size = int(np.random.randint(1, 4))
input_size = int(np.random.randint(1, 4))
hidden_size = int(np.random.randint(1, 4))
num_layers = 4
bias = bool(np.random.choice([True, False]))
dropout = 0

lstm_torch = torch.nn.LSTM(
    input_size, hidden_size, num_layers, bias, batch_first=False, dropout=dropout
)

lstm_man = LSTM(input_size, hidden_size, num_layers, bias=bias, dropout=dropout)

for n, p in lstm_torch.named_parameters():
    weightorbias = n.split("_")[0]
    iorh = n.split("_")[1]
    l = n.split("_")[2][1:]
    
    chunks = torch.chunk(p, 4, dim=0)
    
    vars(vars(lstm_man)[f"cell{l}"])[f"i_{weightorbias}_{iorh}"].data = chunks[0].t().detach().numpy()
    vars(vars(lstm_man)[f"cell{l}"])[f"f_{weightorbias}_{iorh}"].data = chunks[1].t().detach().numpy()
    vars(vars(lstm_man)[f"cell{l}"])[f"g_{weightorbias}_{iorh}"].data = chunks[2].t().detach().numpy()
    vars(vars(lstm_man)[f"cell{l}"])[f"o_{weightorbias}_{iorh}"].data = chunks[3].t().detach().numpy()

xnp = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
x = Tensor(xnp, requires_grad=True)
x_torch = torch.tensor(xnp, requires_grad=True)

out_man, (h_man, c_man) = lstm_man(x)
out_torch, (h_torch, c_torch) = lstm_torch(x_torch)

assert np.allclose(out_man.data, out_torch.detach().numpy()), "output not equal"
assert np.allclose(h_man.data, h_torch.detach().numpy()), "h not equal"
assert np.allclose(c_man.data, c_torch.detach().numpy()), "c not equal"

gradonp = np.random.randn(*out_man.shape).astype(np.float32)
grado = Tensor(gradonp)
grado_torch = torch.tensor(gradonp)

out_man.backward(grado)
out_torch.backward(grado_torch, retain_graph=True)

assert np.allclose(x.grad.data, x_torch.grad.detach().numpy()), "grad from out not equal"

gradhnp = np.random.randn(*h_man.shape).astype(np.float32)
gradh = Tensor(gradhnp)
gradh_torch = torch.tensor(gradhnp)

h_man.backward(gradh)
h_torch.backward(gradh_torch, retain_graph=True)

assert np.allclose(x.grad.data, x_torch.grad.detach().numpy()), "grad from h not equal"

gradcnp = np.random.rand(*c_man.shape).astype(np.float32)
gradc = Tensor(gradcnp)
gradc_torch = torch.tensor(gradcnp)

c_man.backward(gradc)
c_torch.backward(gradc_torch, retain_graph=True)

assert np.allclose(x.grad.data, x_torch.grad.detach().numpy()), "grad from c not equal"



# for _ in range(10):
#     seq_len = int(np.random.randint(1, 10))
#     batch_size = int(np.random.randint(1, 10))
#     input_size = int(np.random.randint(1, 10))
#     hidden_size = int(np.random.randint(1, 10))
#     bias = bool(np.random.choice([True, False]))

#     lstm_man = tocha.nn.LSTMCell(input_size, hidden_size, bias=bias)
#     lstm_torch = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)

#     equate_cell_params(lstm_man, lstm_torch, hidden_size, bias)

#     xnp = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
#     hnp = np.random.randn(batch_size, hidden_size).astype(np.float32)
#     cnp = np.random.randn(batch_size, hidden_size).astype(np.float32)

#     x = Tensor(xnp, requires_grad=True)
#     h = Tensor(hnp)
#     c = Tensor(cnp)

#     x_torch = torch.tensor(xnp, requires_grad=True)
#     h_torch = torch.tensor(hnp)
#     c_torch = torch.tensor(cnp)


#     hp,cp = lstm_man(x[0], (h, c))
#     hpt,cpt = lstm_torch(x_torch[0], (h_torch, c_torch))

#     assert np.allclose(cp.data, cpt.detach().numpy(), atol=1e-6), "Cell state is not equal"
#     assert np.allclose(hp.data, hpt.detach().numpy(), atol=1e-6), "Hidden state is not equal"

#     gradhnp = np.random.randn(*hp.shape)
#     gradh = Tensor(gradhnp)
#     gradh_torch = torch.from_numpy(gradhnp)

#     hp.backward(gradh)
#     hpt.backward(gradh_torch, retain_graph=True)

#     assert np.allclose(x.grad[0].data, x_torch.grad[0].detach().numpy(), atol=1e-6), "Gradient from h is not equal"
#     print(f"Test {_} passed")

#     x.zero_grad()
#     x_torch.grad.zero_()

#     gradcnp = np.random.randn(*cp.shape)
#     gradc = Tensor(gradcnp)
#     gradc_torch = torch.from_numpy(gradcnp)

#     cp.backward(gradc)
#     cpt.backward(gradc_torch)

#     assert np.allclose(x.grad[0].data, x_torch.grad[0].detach().numpy(), atol=1e-6), "Gradient from c is not equal"
