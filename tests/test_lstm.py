import unittest
import tocha
import tocha.nn
import torch
import numpy as np


class TestLSTMCell(unittest.TestCase):
    def test_rnn_against_torch_output_0(self):
        np.random.seed(0)
        torch.manual_seed(0)
        for _ in range(5): # tested for 100 successfully before
            seq_len = int(np.random.randint(1, 4))
            batch_size = int(np.random.randint(1, 4))
            input_size = int(np.random.randint(1, 4))
            hidden_size = int(np.random.randint(1, 4))
            num_layers = int(np.random.randint(1, 4))
            bias = bool(np.random.choice([True, False]))
            dropout = 0 # if dropout is not zero, randomness will make the test fail

            lstm_torch = torch.nn.LSTM(
                input_size, hidden_size, num_layers, bias, batch_first=False, dropout=dropout
            )
            lstm_man = tocha.nn.LSTM(input_size, hidden_size, num_layers, bias=bias, dropout=dropout)

            # copy weights
            for n, p in lstm_torch.named_parameters():
                weightorbias = n.split("_")[0]
                iorh = n.split("_")[1]
                l = n.split("_")[2][1:]
                
                chunks = torch.chunk(p, 4, dim=0)
                
                vars(vars(lstm_man)[f"cell{l}"])[f"i_{weightorbias}_{iorh}"].data = chunks[0].t().detach().numpy().copy()
                vars(vars(lstm_man)[f"cell{l}"])[f"f_{weightorbias}_{iorh}"].data = chunks[1].t().detach().numpy().copy()
                vars(vars(lstm_man)[f"cell{l}"])[f"g_{weightorbias}_{iorh}"].data = chunks[2].t().detach().numpy().copy()
                vars(vars(lstm_man)[f"cell{l}"])[f"o_{weightorbias}_{iorh}"].data = chunks[3].t().detach().numpy().copy()

            # forward
            xnp = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
            x = tocha.Tensor(xnp, requires_grad=True)
            x_torch = torch.tensor(xnp, requires_grad=True)

            out_man, (h_man, c_man) = lstm_man(x)
            out_torch, (h_torch, c_torch) = lstm_torch(x_torch)

            assert np.allclose(out_man.data, out_torch.detach().numpy()), "output not equal"

            # backward
            gradonp = np.random.randn(*out_man.shape).astype(np.float32)
            grado = tocha.Tensor(gradonp)
            grado_torch = torch.tensor(gradonp)

            out_man.backward(grado)
            out_torch.backward(grado_torch, retain_graph=True)

            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy()), "grad from out not equal"

            gradhnp = np.random.randn(*h_man.shape).astype(np.float32)
            gradh = tocha.Tensor(gradhnp)
            gradh_torch = torch.tensor(gradhnp)

            h_man.backward(gradh)
            h_torch.backward(gradh_torch, retain_graph=True)

            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-5), "grad from h not equal"

            gradcnp = np.random.rand(*c_man.shape).astype(np.float32)
            gradc = tocha.Tensor(gradcnp)
            gradc_torch = torch.tensor(gradcnp)

            c_man.backward(gradc)
            c_torch.backward(gradc_torch, retain_graph=True)

            assert np.allclose(x.grad.data, x_torch.grad.detach().numpy(), atol=1e-5), "grad from c not equal"


