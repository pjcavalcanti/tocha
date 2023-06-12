import torch
import numpy as np

import tocha
from tocha import Tensor
from tocha.module import Parameter, Module

nhead = 2
d_model = 3 * nhead
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 5
dropout = 0.0
layer_norm_eps = 1e-5
batch_first = True

batch_size = 3
seq_len = 3


trans_torch = torch.nn.Transformer(
    d_model=d_model,
    nhead=nhead,
    num_decoder_layers=num_encoder_layers,
    num_encoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    layer_norm_eps=layer_norm_eps,
    batch_first=batch_first,
)

enc_layer_torch = torch.nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    layer_norm_eps=layer_norm_eps,
    batch_first=batch_first,
)
dec_layer_torch = torch.nn.TransformerDecoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    layer_norm_eps=layer_norm_eps,
    batch_first=batch_first,
)
enc_torch = torch.nn.TransformerEncoder(encoder_layer=enc_layer_torch, num_layers=num_encoder_layers)
dec_torch = torch.nn.TransformerDecoder(decoder_layer=dec_layer_torch, num_layers=num_decoder_layers)

for name_enc, param_enc in enc_torch.named_parameters():
    # 'layers' is a part of the parameter name in enc_torch
    # replace 'layers' with 'encoder.layers' for the transformer
    name_trans = name_enc.replace("layers", "encoder.layers")
    

    # get the parameter from the transformer
    param_trans = trans_torch
    for part in name_trans.split('.'):
        param_trans = getattr(param_trans, part)

    # update the parameter in enc_torch
    param_enc.data = param_trans.data.clone()
    
for name_enc, param_enc in dec_torch.named_parameters():
    name_trans = name_enc.replace("layers", "decoder.layers")
    
    param_trans = trans_torch
    for part in name_trans.split('.'):
        param_trans = getattr(param_trans, part)
    
    param_enc.data = param_trans.data.clone()
    

src = torch.randn(batch_size, seq_len, d_model)
tgt = torch.randn(batch_size, seq_len, d_model)
mem_man = enc_torch(src)
out_man = dec_torch(tgt, mem_man)

out_torch = trans_torch(src, tgt)

print(torch.allclose(out_man, out_torch, atol=1e-5))

