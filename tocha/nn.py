import tocha
import tocha.functional as F
from tocha.module import Module, Parameter
from autograd.tensor import Tensor, Arrayable, ensure_array

import numpy as np
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import copy

## Non-linearities


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, dim=self.axis)


## Basic layers


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = Parameter(
            np.random.randn(in_features, out_features) / np.sqrt(in_features)
        )
        self.bias = Parameter(np.random.randn(out_features) / np.sqrt(in_features))

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weights + self.bias
        return out


class Conv2d(Module):
    # TODO: Add padding and stride
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Tuple[int, ...],
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.bias = None

        self.weight = Parameter(
            np.random.randn(out_features, in_features, kernel_size[0] * kernel_size[1])
            / np.sqrt(in_features)
        )
        if bias:
            self.bias = Parameter(np.random.randn(out_features) / np.sqrt(in_features))

    def forward(self, x: Tensor) -> Tensor:
        assert (
            len(x.shape) == 4
        ), "Input tensor must be (batch_size, channels, height, width)"
        # Separate submatrices with im2col
        out = F.im2col(x, self.kernel_size)
        # Apply convolution
        #  out    = # (B, Cin, k1*k2, (H - k1 + 1)*(W - k2 + 1))
        #  weight = # (Cout, Cin, k1*k2)
        out = tocha.tensordot(out, self.weight, axes=((-3, -2), (1, 2)))

        # Add bias
        # out =  # (B, (H - k1 + 1)*(W - k2 + 1),  Cout)
        # bias = # (Cout,)
        if self.bias is not None:
            out = out + self.bias

        # Transpose to get the right index order
        axes = tuple(range(len(out.shape)))
        axes = axes[:-2] + (axes[-1], axes[-2])
        out = tocha.transpose(out, axes=axes)
        # out = (B, Cout, (H - k1 + 1)*(W - k2 + 1))

        # Reshape to get the right output shape
        batch = x.shape[0]
        x_out = x.shape[-2] - self.kernel_size[0] + 1
        y_out = x.shape[-1] - self.kernel_size[1] + 1
        out = out.reshape((batch, self.out_features, x_out, y_out))
        return out


## Containers


class Sequential(Module):
    def __init__(self, layers: List[Module]) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        for l, layer in enumerate(self.layers):
            for name, param in layer.named_parameters():
                yield f"{l}.{name}", param

    def named_modules(self) -> Iterator[Tuple[str, Parameter]]:
        for l, layer in enumerate(self.layers):
            yield f"{l}", layer
            for name, module in layer.named_modules():
                yield f"{l}.{name}", module

    def __getitem__(self, i):
        assert isinstance(i, int), "Index must be an integer"
        return self.layers[i]


## Regularization layers


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p  # probability of dropping a number

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask_np = np.random.binomial(1, 1 - self.p, x.shape)
            mask = tocha.tensor(mask_np, requires_grad=False)
            return mask * x / (1 - self.p)
        else:
            return x


class BatchNorm1d(Module):
    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        # here i use dim rather than num_features, which deviates from pytorch
        # this is because i want to avoid reshape in forward
        # so i can declare gamma and beta with the right shape in init
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.adapted = False

        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))

        self.running_mean = Tensor(0.0, requires_grad=False)
        self.running_var = Tensor(1.0, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        assert (
            len(x.shape) == 2 or len(x.shape) == 3
        ), "Input tensor must be (batch_size, num_features) or (batch_size, num_features, seq_len)"
        # reshape gamma and beta if necessary
        # this can cause bugs if the user changes the shape of the input
        self._adapt_shapes(x)

        if self.training:
            # the docs claim the var is biased, but actually it is unbiased only in the forward pass
            # for the running var, the var is unbiased
            mean = x.mean(axis=self.axis, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=self.axis, keepdims=True)

            n = x.shape[0] * x.shape[2] if len(x.shape) == 3 else x.shape[0]
            self.running_mean = Tensor(
                (1 - self.momentum) * self.running_mean.data
                + self.momentum * mean.data,
                requires_grad=False,
            )
            self.running_var = Tensor(
                (1 - self.momentum) * self.running_var.data
                + self.momentum * var.data * n / (n - 1),
                requires_grad=False,
            )
        else:
            mean = self.running_mean
            var = self.running_var

        out = (x - mean) / (var + self.eps) ** 0.5
        out = self.gamma * out + self.beta
        return out

    def _adapt_shapes(self, x: Tensor) -> None:
        assert (
            len(x.shape) == 2 or len(x.shape) == 3
        ), "Expected 2D or 3D input (got {}D input)".format(len(x.shape))

        if len(self.gamma.shape) != len(x.shape) and not self.adapted:
            if len(x.shape) == 2:
                self.gamma = self.gamma.reshape((1, self.num_features))
                self.beta = self.beta.reshape((1, self.num_features))
            elif len(x.shape) == 3:
                self.gamma = self.gamma.reshape((1, self.num_features, 1))
                self.beta = self.beta.reshape((1, self.num_features, 1))
            self.adapted = True

        if len(x.shape) == 2:
            self.axis = 0
        elif len(x.shape) == 3:
            self.axis = (0, 2)


class BatchNorm2d(Module):
    def __init__(
        self, n_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter(np.ones((1, n_features, 1, 1)))
        self.beta = Parameter(np.zeros((1, n_features, 1, 1)))

        self.running_mean = Tensor(np.zeros((1, n_features, 1, 1)), requires_grad=False)
        self.running_var = Tensor(np.ones((1, n_features, 1, 1)), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        assert (
            len(x.shape) == 4
        ), "Input tensor must be (batch_size, num_features, height, width)"
        axis = (0, 2, 3)
        if self.training:
            mean = x.mean(axis=axis, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=axis, keepdims=True)

            n = x.shape[0] * x.shape[2] * x.shape[3]

            self.running_mean = Tensor(
                (1 - self.momentum) * self.running_mean.data
                + self.momentum * mean.data,
                requires_grad=False,
            )
            self.running_var = Tensor(
                (1 - self.momentum) * self.running_var.data
                + self.momentum * var.data * n / (n - 1),
                requires_grad=False,
            )
        else:
            mean = self.running_mean
            var = self.running_var
        out = (x - mean) / (var + self.eps) ** 0.5
        out = self.gamma * out + self.beta
        return out


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[List[int], int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (
            [normalized_shape]
            if isinstance(normalized_shape, int)
            else normalized_shape
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape))
            self.bias = Parameter(np.zeros(self.normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(len(x.shape) - i for i in range(len(self.normalized_shape), 0, -1))
        mean = x.mean(dims, keepdims=True)
        var = ((x - mean) ** 2).mean(dims, keepdims=True)
        out = (x - mean) / (var + self.eps).sqrt()
        if self.elementwise_affine:
            out = self.weight * out + self.bias
        return out


## Recurrent Layers


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
            new_cell = RNNCell(
                input_size if l == 0 else hidden_size,
                hidden_size,
                bias,
                nonlinearity,
                dtype,
            )
            self.register_module(f"cell_{l}", new_cell)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        # I won't use single tensors due to how autograd works
        # Initialize hidden states for all time steps and layers
        h = [
            Tensor(
                np.zeros((1, batch_size, self.hidden_size), dtype=self.dtype),
                requires_grad=False,
            )
            for _ in range(self.num_layers)
        ]
        # Initialize outputs for all time steps
        outputs = [
            Tensor(
                np.zeros((1, batch_size, self.hidden_size), dtype=self.dtype),
                requires_grad=False,
            )
            for _ in range(sequence_length)
        ]
        for t in range(sequence_length):
            x_in = x[t]
            for c, cell in enumerate(self.children()):
                h[c] = cell(x_in, h[c])

                if (
                    self.dropout is not None
                    and self.training
                    and c < self.num_layers - 1
                ):
                    h[c] = self.dropout(h[c])

                x_in = h[c]
            outputs[t] = h[-1]
        outputs = tocha.concatenate(outputs, axis=0)
        h = tocha.concatenate(h, axis=0)
        return (outputs, h)

    def children(self) -> Iterator[Module]:
        for child in super().children():
            if isinstance(child, Dropout):
                continue
            yield child


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bias = bias

        self.i_weight_ih = Parameter(
            np.random.randn(input_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.i_weight_hh = Parameter(
            np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.f_weight_ih = Parameter(
            np.random.randn(input_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.f_weight_hh = Parameter(
            np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.g_weight_ih = Parameter(
            np.random.randn(input_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.g_weight_hh = Parameter(
            np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.o_weight_ih = Parameter(
            np.random.randn(input_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.o_weight_hh = Parameter(
            np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )

        if self.bias:
            self.i_bias_ih = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.i_bias_hh = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.f_bias_ih = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.f_bias_hh = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.g_bias_ih = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.g_bias_hh = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.o_bias_ih = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )
            self.o_bias_hh = Parameter(
                np.random.randn(hidden_size) / np.sqrt(hidden_size)
            )

    def forward(self, x: Tensor, hc: Tuple[Tensor, Tensor]) -> Tensor:
        h, c = hc

        pre_i = x @ self.i_weight_ih + h @ self.i_weight_hh
        pre_f = x @ self.f_weight_ih + h @ self.f_weight_hh
        pre_g = x @ self.g_weight_ih + h @ self.g_weight_hh
        pre_o = x @ self.o_weight_ih + h @ self.o_weight_hh
        if self.bias:
            pre_i += self.i_bias_ih + self.i_bias_hh
            pre_f += self.f_bias_ih + self.f_bias_hh
            pre_g += self.g_bias_ih + self.g_bias_hh
            pre_o += self.o_bias_ih + self.o_bias_hh
        i = F.sigmoid(pre_i)
        f = F.sigmoid(pre_f)
        g = F.tanh(pre_g)
        o = F.sigmoid(pre_o)

        cp = f * c + i * g
        hp = o * F.tanh(cp)
        return hp, cp


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
            new_cell = LSTMCell(
                input_size if i == 0 else hidden_size, hidden_size, bias=bias
            )
            self.register_module(f"cell{i}", new_cell)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        hs = [
            Tensor(np.zeros((1, batch_size, self.hidden_size)))
            for _ in range(self.num_layers)
        ]
        cs = [
            Tensor(np.zeros((1, batch_size, self.hidden_size)))
            for _ in range(self.num_layers)
        ]

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


class ScaledDotProductAttentionHead(Module):
    # Note the projection is absorbed here
    def __init__(self, embed_dim: int, head_dim: int, bias: bool) -> None:
        self.head_dim = head_dim
        self.num_heads = embed_dim // head_dim
        self.bias = bias
        self.scale = np.sqrt(head_dim)

        self.q_proj_weight = Parameter(np.random.randn(embed_dim, head_dim))
        self.k_proj_weight = Parameter(np.random.randn(embed_dim, head_dim))
        self.v_proj_weight = Parameter(np.random.randn(embed_dim, head_dim))

        if self.bias:
            self.q_proj_bias = Parameter(np.random.randn(head_dim))
            self.k_proj_bias = Parameter(np.random.randn(head_dim))
            self.v_proj_bias = Parameter(np.random.randn(head_dim))

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, att_mask: Optional[Tensor] = None
    ) -> Tensor:
        # use attention formula from https://arxiv.org/pdf/1706.03762.pdf
        Q = q @ self.q_proj_weight
        K = k @ self.k_proj_weight
        V = v @ self.v_proj_weight
        if self.bias:
            Q += self.q_proj_bias
            K += self.k_proj_bias
            V += self.v_proj_bias

        att = Q @ K.transpose((0, 2, 1)) / self.scale
        if att_mask is not None:
            att = att_mask * att
        att = F.softmax(att, dim=-1) @ V
        return att


class MultiheadAttention(Module):
    # Assumes that the input has batch_first = True
    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert dropout >= 0.0 and dropout <= 1.0, "dropout must be between 0.0 and 1.0"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

        self.head_dim = embed_dim // num_heads

        for i in range(num_heads):
            new_head = ScaledDotProductAttentionHead(embed_dim, self.head_dim, bias)
            self.register_module(f"head_{i}", new_head)
        self.out_proj_weight = Parameter(np.random.randn(embed_dim, embed_dim))
        if self.bias:
            self.out_proj_bias = Parameter(np.random.randn(embed_dim))

        self.dropout = Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        # apply all heads
        head_outputs = [
            getattr(self, f"head_{i}")(q, k, v, attn_mask)
            for i in range(self.num_heads)
        ]
        # concatenate head outputs, then project
        output = tocha.concatenate(head_outputs, axis=-1) @ self.out_proj_weight
        if self.bias:
            output += self.out_proj_bias
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.layer_norm_eps = layer_norm_eps
        self.dim_feedforwad = dim_feedforward
        self.layer_norm_eps = layer_norm_eps

        self.self_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Attention, dropout, norm
        out = self.self_attn(x, x, x)
        out = self.dropout(out)
        out = self.norm1(x + out)
        # Feedforward
        out_ff = self.linear1(out)
        out_ff = self.dropout1(out_ff)  # not sure if this dropout should be here
        out_ff = F.relu(out_ff)
        out_ff = self.linear2(out_ff)
        # Dropout, norm
        out_ff = self.dropout2(out_ff)
        out = self.norm2(out + out_ff)
        return out


class TransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert dropout >= 0 and dropout <= 1, "dropout must be between 0 and 1"

        self.d_model = d_model
        self.nhead = nhead
        self.layer_norm_eps = layer_norm_eps
        self.dim_feedforward = dim_feedforward
        self.head_dim = d_model // nhead

        self.self_attn = MultiheadAttention(d_model, nhead)
        self.multihead_attn = MultiheadAttention(d_model, nhead)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout = Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        # First attention on input
        out1 = self.self_attn(tgt, tgt, tgt)
        out1 = self.dropout(out1)
        out1 = self.norm1(out1 + tgt)

        # Second attention on encoder output and output of first attention
        out2 = self.multihead_attn(out1, memory, memory)
        out2 = self.dropout(out2)
        out2 = self.norm2(out2 + out1)

        # Feedforward
        out3 = self.linear1(out2)
        out3 = self.dropout(out3)
        out3 = F.relu(out3)
        out3 = self.linear2(out3)
        out4 = self.dropout(out3)
        out4 = self.norm3(out4 + out2)

        return out4


class TransformerEncoder(Module):
    # Like in torch, all layers start with the same parameters
    # This is unexpected, but it makes it easier to test against torch
    def __init__(
        self,
        encoder_layer: Iterable[TransformerEncoderLayer],
        num_layers: int,
        norm: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.layers = [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        self.norm = (
            None
            if norm is None
            else LayerNorm(encoder_layer.d_model, eps=encoder_layer.layer_norm_eps)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoder(Module):
    # Like in torch, all layers start with the same parameters
    # This is unexpected, but it makes it easier to test against torch
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.layers = [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        self.norm = (
            None
            if norm is None
            else LayerNorm(decoder_layer.d_model, eps=decoder_layer.layer_norm_eps)
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        out = tgt
        for layer in self.layers:
            out = layer(out, memory)
        if self.norm is not None:
            out = self.norm(out)
        return out

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