import torch
import torch.nn as nn

input_size = 5
hidden_size = 10
num_layers = 1
nonlinearity = 'relu'
bias=False
dropout=0
bidirectional=False
batch_first =True

torch.manual_seed(0)
rnn = nn.RNN(input_size=input_size,
             hidden_size=hidden_size,
             num_layers=num_layers,
             nonlinearity=nonlinearity,
             bias=bias,
             dropout=dropout,
             bidirectional=bidirectional,
             batch_first=False)

B = 3
T = 2
x = torch.randn(T, B, input_size)
h = torch.randn(T, B, hidden_size)

for name, param in rnn.named_parameters():
    print(f"{name=}, {param.shape=}")

# for i in x.shape
out_man = torch.relu(x @ rnn.weight_ih_l0.t())
out = rnn(x)
for outi in out:
    print(f"{outi.shape=}")

print(out_man.shape)
print(out_man, out[0], sep="\n")