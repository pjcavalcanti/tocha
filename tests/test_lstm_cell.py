import unittest
import tocha
import torch
import numpy as np


class TestLSTMCell(unittest.TestCase):
    def test_rnn_against_torch_output_0(self):
        np.random.seed(0)
        for _ in range(100):
            seq_len = int(np.random.randint(1, 10))
            batch_size = int(np.random.randint(1, 10))
            input_size = int(np.random.randint(1, 10))
            hidden_size = int(np.random.randint(1, 10))
            bias = bool(np.random.choice([True, False]))

            lstm_man = tocha.nn.LSTMCell(input_size, hidden_size, bias=bias)
            lstm_torch = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)

            weight_ih = lstm_torch.weight_ih
            weight_hh = lstm_torch.weight_hh
            bias_ih = lstm_torch.bias_ih
            bias_hh = lstm_torch.bias_hh

            lstm_man.i_weight_ih.data = weight_ih[:hidden_size, :].t().detach().numpy()
            lstm_man.i_weight_hh.data = weight_hh[:hidden_size, :].t().detach().numpy()
            lstm_man.f_weight_ih.data = weight_ih[hidden_size : 2 * hidden_size, :].t().detach().numpy()
            lstm_man.f_weight_hh.data = weight_hh[hidden_size : 2 * hidden_size, :].t().detach().numpy()
            lstm_man.g_weight_ih.data = weight_ih[2 * hidden_size : 3 * hidden_size, :].t().detach().numpy()
            lstm_man.g_weight_hh.data = weight_hh[2 * hidden_size : 3 * hidden_size, :].t().detach().numpy()
            lstm_man.o_weight_ih.data = weight_ih[3 * hidden_size :, :].t().detach().numpy()
            lstm_man.o_weight_hh.data = weight_hh[3 * hidden_size :, :].t().detach().numpy()
            if bias:
                lstm_man.i_bias_ih.data = bias_ih[:hidden_size].t().detach().numpy()
                lstm_man.i_bias_hh.data = bias_hh[:hidden_size].t().detach().numpy()
                lstm_man.f_bias_ih.data = bias_ih[hidden_size : 2 * hidden_size].t().detach().numpy()
                lstm_man.f_bias_hh.data = bias_hh[hidden_size : 2 * hidden_size].t().detach().numpy()
                lstm_man.g_bias_ih.data = bias_ih[2 * hidden_size : 3 * hidden_size].t().detach().numpy()
                lstm_man.g_bias_hh.data = bias_hh[2 * hidden_size : 3 * hidden_size].t().detach().numpy()
                lstm_man.o_bias_ih.data = bias_ih[3 * hidden_size :].t().detach().numpy()
                lstm_man.o_bias_hh.data = bias_hh[3 * hidden_size :].t().detach().numpy()

            xnp = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
            hnp = np.random.randn(batch_size, hidden_size).astype(np.float32)
            cnp = np.random.randn(batch_size, hidden_size).astype(np.float32)

            x = tocha.Tensor(xnp, requires_grad=True)
            h = tocha.Tensor(hnp)
            c = tocha.Tensor(cnp)

            x_torch = torch.tensor(xnp, requires_grad=True)
            h_torch = torch.tensor(hnp)
            c_torch = torch.tensor(cnp)
                    

            hp,cp = lstm_man(x[0], (h, c))
            hpt,cpt = lstm_torch(x_torch[0], (h_torch, c_torch))

            assert np.allclose(cp.data, cpt.detach().numpy(), atol=1e-6), "Cell state is not equal"
            assert np.allclose(hp.data, hpt.detach().numpy(), atol=1e-6), "Hidden state is not equal"

            gradhnp = np.random.randn(*hp.shape)
            gradh = tocha.Tensor(gradhnp)
            gradh_torch = torch.from_numpy(gradhnp)

            hp.backward(gradh)
            hpt.backward(gradh_torch, retain_graph=True)

            assert np.allclose(x.grad[0].data, x_torch.grad[0].detach().numpy(), atol=1e-6), "Gradient from h is not equal"
            print(f"Test {_} passed")
            
            x.zero_grad()
            x_torch.grad.zero_()
            
            gradcnp = np.random.randn(*cp.shape)
            gradc = tocha.Tensor(gradcnp)
            gradc_torch = torch.from_numpy(gradcnp)
            
            cp.backward(gradc)
            cpt.backward(gradc_torch)
            
            assert np.allclose(x.grad[0].data, x_torch.grad[0].detach().numpy(), atol=1e-6), "Gradient from c is not equal"
            