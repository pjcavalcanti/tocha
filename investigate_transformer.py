import torch
import numpy as np

import tocha
from tocha import Tensor
from tocha.module import Parameter, Module
from tocha.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

def check_torch_manually():
    torch.manual_seed(0)
    for _ in range(50):       
        nhead = np.random.randint(1, 5)
        d_model = np.random.randint(1, 5) * nhead
        num_encoder_layers = np.random.randint(1, 5)
        num_decoder_layers = np.random.randint(1, 5)
        dim_feedforward = np.random.randint(1, 5)
        dropout = 0.0
        layer_norm_eps = np.random.randint(1, 5) * 10 **(-np.random.randint(1, 5))
        batch_first = True

        batch_size = np.random.randint(1, 5)
        seq_len = np.random.randint(1, 5)

        trans_torch = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            dtype=torch.float64,
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
        enc_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, dtype=torch.float64)
        enc_torch = torch.nn.TransformerEncoder(
            encoder_layer=enc_layer_torch, num_layers=num_encoder_layers, norm=enc_norm
        )
        dec_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, dtype=torch.float64)
        dec_torch = torch.nn.TransformerDecoder(
            decoder_layer=dec_layer_torch, num_layers=num_decoder_layers, norm=dec_norm
        )

        for l, layer in enumerate(enc_torch.layers):
            for n, p in layer.named_parameters():
                param_man = layer
                param_tor = trans_torch.encoder.layers[l]
                for part in n.split("."):
                    param_man = getattr(param_man, part)
                    param_tor = getattr(param_tor, part)
                param_man.data = param_tor.data.clone()
        for l, layer in enumerate(dec_torch.layers):
            for n, p in layer.named_parameters():
                param_man = layer
                param_tor = trans_torch.decoder.layers[l]
                for part in n.split("."):
                    param_man = getattr(param_man, part)
                    param_tor = getattr(param_tor, part)
                param_man.data = param_tor.data.clone()

        src = torch.randn(batch_size, seq_len, d_model, dtype=torch.float64)
        tgt = torch.randn(batch_size, seq_len, d_model, dtype=torch.float64)
        mem_man = enc_torch(src)
        out_man = dec_torch(tgt, mem_man)

        out_torch = trans_torch(src, tgt)

        print(torch.allclose(out_man, out_torch, atol=1e-10))


def equate_tocha_to_torch_linear(toch, torc):
    toch.weights.data = torc.weight.T.detach().numpy()
    toch.bias.data = torc.bias.detach().numpy()


def equate_tocha_to_torch_layer_norm(toch, torc):
    toch.weight.data = torc.weight.detach().numpy()
    toch.bias.data = torc.bias.detach().numpy()


def equate_tocha_to_torch_attention(toch, torc, bias, num_heads):
    qkv_weight = torc.in_proj_weight
    out_weight = torc.out_proj.weight
    if bias:
        qkv_bias = torc.in_proj_bias
        out_bias = torc.out_proj.bias

    # separate q,k,v
    q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
    if bias:
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

    # separate head weights/biases
    head_wq = q_weight.t().chunk(dim=1, chunks=num_heads)
    head_wk = k_weight.t().chunk(dim=1, chunks=num_heads)
    head_wv = v_weight.t().chunk(dim=1, chunks=num_heads)
    if bias:
        head_bq = q_bias.chunk(dim=0, chunks=num_heads)
        head_bk = k_bias.chunk(dim=0, chunks=num_heads)
        head_bv = v_bias.chunk(dim=0, chunks=num_heads)

    for h, (wq, wk, wv) in enumerate(zip(head_wq, head_wk, head_wv)):
        head = getattr(toch, f"head_{h}")
        head.q_proj_weight.data = wq.detach().numpy()
        head.k_proj_weight.data = wk.detach().numpy()
        head.v_proj_weight.data = wv.detach().numpy()
    if bias:
        for h, (bq, bk, bv) in enumerate(zip(head_bq, head_bk, head_bv)):
            head = getattr(toch, f"head_{h}")
            head.q_proj_bias.data = bq.detach().numpy()
            head.k_proj_bias.data = bk.detach().numpy()
            head.v_proj_bias.data = bv.detach().numpy()
    toch.out_proj_weight.data = out_weight.t().detach().numpy()
    if bias:
        toch.out_proj_bias.data = out_bias.detach().numpy()


def equate_tocha_to_torch_transformer_decoder_layer(toch, torc, nhead):
    equate_tocha_to_torch_attention(
        toch.self_attn, torc.self_attn, bias=True, num_heads=nhead
    )
    equate_tocha_to_torch_attention(
        toch.multihead_attn, torc.multihead_attn, bias=True, num_heads=nhead
    )
    equate_tocha_to_torch_linear(toch.linear1, torc.linear1)
    equate_tocha_to_torch_linear(toch.linear2, torc.linear2)
    equate_tocha_to_torch_layer_norm(toch.norm1, torc.norm1)
    equate_tocha_to_torch_layer_norm(toch.norm2, torc.norm2)
    equate_tocha_to_torch_layer_norm(toch.norm3, torc.norm3)

def equate_tocha_to_torch_transformer_encoder_layer(toch, torc):
    equate_tocha_to_torch_attention(toch.self_attn, torc.self_attn, True, toch.nhead)

    toch.linear1.weights.data = torc.linear1.weight.T.detach().numpy()
    toch.linear1.bias.data = torc.linear1.bias.detach().numpy()
    toch.linear2.weights.data = torc.linear2.weight.T.detach().numpy()
    toch.linear2.bias.data = torc.linear2.bias.detach().numpy()
    toch.norm1.weight.data = torc.norm1.weight.detach().numpy()
    toch.norm1.bias.data = torc.norm1.bias.detach().numpy()
    toch.norm2.weight.data = torc.norm2.weight.detach().numpy()
    toch.norm2.bias.data = torc.norm2.bias.detach().numpy()


class Transformer(Module):
    def __init__(
        self,
        d_model: float,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm=True)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, norm=True)
        
        # randomize parameter initializations
        for layer in [*self.encoder.layers, *self.decoder.layers]:
            for n, p in layer.named_parameters():
                parts = n.split(".")
                param = getattr(layer, parts[0])
                for part in parts[1:]:
                    param = getattr(param, part)
                param.data = np.random.randn(*param.shape).astype(np.float32)
                
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        mem = self.encoder(src)
        out = self.decoder(tgt, mem)
        return out
    
def check_tocha_against_torch_forward():
    np.random.seed(0)
    torch.manual_seed(0)
    for _ in range(100):        
        nhead = np.random.randint(1, 5)
        d_model = np.random.randint(1, 5) * nhead
        num_encoder_layers = np.random.randint(1, 5)
        num_decoder_layers = np.random.randint(1, 5)
        dim_feedforward = np.random.randint(1, 5)
        layer_norm_eps = np.random.randint(1, 5) * 10 ** (-np.random.randint(1, 5))
        dropout = 0.0
        batch_first = True

        batch_size = np.random.randint(1, 5)
        seq_len = np.random.randint(1, 5)
                    
        trans_torch = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            dtype = torch.float64,
        )
        trans_tocha = Transformer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            num_decoder_layers=num_decoder_layers,
            num_encoder_layers=num_encoder_layers,
        )

        enc_tocha = trans_tocha.encoder
        dec_tocha = trans_tocha.decoder
        # print(f"enc layers: {len(enc_tocha.layers)}, {len(trans_torch.encoder.layers)}")
        for l, layer in enumerate(enc_tocha.layers):
            # print(l)
            equate_tocha_to_torch_transformer_encoder_layer(layer, trans_torch.encoder.layers[l])
        for l, layer in enumerate(dec_tocha.layers):
            # print(l)
            equate_tocha_to_torch_transformer_decoder_layer(layer, trans_torch.decoder.layers[l], nhead)

        srcnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)
        tgtnp = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)
        src_tocha = tocha.tensor(srcnp, requires_grad=True)
        src_torch = torch.tensor(srcnp, requires_grad=True, dtype=torch.float64)
        tgt_tocha = tocha.tensor(tgtnp, requires_grad=True)
        tgt_torch = torch.tensor(tgtnp, requires_grad=True, dtype=torch.float64)

        out_tocha = trans_tocha(src_tocha, tgt_tocha)
        out_torch = trans_torch(src_torch, tgt_torch)

        passforward = np.allclose(out_tocha.data, out_torch.detach().numpy(), atol=1e-10)
        assert passforward, f"forward pass failed"

check_tocha_against_torch_forward()