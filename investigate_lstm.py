import torch
import tocha
import tocha.nn as nn
import tocha.functional as F

lstm_torch = torch.nn.LSTMCell(3, 5)
full_lstm = torch.nn.LSTM(3, 5, batch_first=False)

for name, p in lstm_torch.named_parameters():
    print(name, p.shape)
print("\n\n")
for name, p in full_lstm.named_parameters():
    print(name, p.shape)