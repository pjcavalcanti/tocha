import tocha
import tocha.functional as F
from tocha.module import Module, Parameter
from autograd.tensor import Tensor, Arrayable, ensure_array

import numpy as np
from typing import List, Tuple
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
            new_cell = RNNCell(input_size if l==0 else hidden_size, hidden_size, bias, nonlinearity, dtype)                
            self.register_module(f"cell_{l}", new_cell)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        time_length = x.shape[0]
        batch_size = x.shape[1]

        # Initialize hidden states for all time steps and layers
        hidden_states = [
            [
                Tensor(
                    np.zeros((1, batch_size, self.hidden_size), dtype=self.dtype),
                    requires_grad=False,
                )
                for _ in range(self.num_layers)
            ]
            for _ in range(time_length)
        ]
        # Initialize outputs for all time steps
        outputs = []
        for time in range(time_length):
            for c, cell in enumerate(self.children()):
                if c == 0:
                    hidden_states[time][c] = cell(x[time], hidden_states[time - 1][c])
                else:
                    hidden_states[time][c] = cell(
                        hidden_states[time][c - 1], hidden_states[time - 1][c]
                    )
                if self.dropout is not None and self.training and c < self.num_layers - 1:
                    hidden_states[time][c] = self.dropout(hidden_states[time][c])
            # After all layers have been applied, save the output of the last layer
            outputs.append(hidden_states[time][-1])
        outputs = tocha.concatenate(outputs, axis=0)  #
        hidden_states = tocha.concatenate(hidden_states[-1], axis=0)
        return (outputs, hidden_states)






# class RNN(Module):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         num_layers: int,
#         nonlinearity: str,
#         bias: bool,
#         dropout: float = 0.0,
#         dtype: str = "float32",
#     ) -> None:
#         assert (
#             input_size > 0
#             and hidden_size > 0
#             and num_layers > 0
#             and nonlinearity in ["relu", "tanh"]
#             and dropout >= 0.0
#             and dropout <= 1.0
#         )
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bias = bias
#         self.dropout = Dropout(p=dropout) if dropout > 0.0 else None
#         self.activation = F.relu if nonlinearity == "relu" else F.tanh
#         self.dtype = np.float32 if dtype == "float32" else np.float64

#         for l in range(self.num_layers):
#             self.register_parameter(
#                 f"weight_ih_l{l}",
#                 Parameter(
#                     np.random.randn(hidden_size, input_size if l == 0 else hidden_size)
#                     / np.sqrt(hidden_size),
#                     name=f"weight_ih_l{l}",
#                 ),
#             )
#             self.register_parameter(
#                 f"weight_hh_l{l}",
#                 Parameter(
#                     np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size),
#                     name=f"weight_hh_l{l}",
#                 ),
#             )
#             if self.bias:
#                 self.register_parameter(
#                     f"bias_ih_l{l}",
#                     Parameter(
#                         np.random.randn(hidden_size) / np.sqrt(hidden_size),
#                         name=f"bias_ih_l{l}",
#                     ),
#                 )
#                 self.register_parameter(
#                     f"bias_hh_l{l}",
#                     Parameter(
#                         np.random.randn(hidden_size) / np.sqrt(hidden_size),
#                         name=f"bias_hh_l{l}",
#                     ),
#                 )

#     def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         time_length = x.shape[0]
#         batch_size = x.shape[1]

#         # Initialize hidden states for all time steps and layers
#         hidden_states = [
#             [
#                 Tensor(
#                     np.zeros((1, batch_size, self.hidden_size), dtype=self.dtype),
#                     requires_grad=False,
#                 )
#                 for _ in range(self.num_layers)
#             ]
#             for _ in range(time_length)
#         ]
#         # Initialize outputs for all time steps
#         outputs = []
#         for time in range(time_length):
#             # Apply the layers on the input of the current time step
#             for layer in range(self.num_layers):
#                 # Apply the layer
#                 if layer == 0:
#                     hidden_states[time][layer] = self.apply_layer_i(
#                         layer, x[time], hidden_states[time - 1][layer]
#                     )
#                 else:
#                     hidden_states[time][layer] = self.apply_layer_i(
#                         layer, hidden_states[time][layer - 1], hidden_states[time - 1][layer]
#                     )
#             # After all layers have been applied, save the output of the last layer
#             outputs.append(hidden_states[time][-1])
#         outputs = tocha.concatenate(outputs, axis=0) # 
#         hidden_states = tocha.concatenate(hidden_states[-1], axis=0)
#         return (outputs, hidden_states)

#     def apply_layer_i(self, l: int, x_in: Tensor, h_in: Tensor):
#         # Apply the weights
#         weight_ih = vars(self)[f"weight_ih_l{l}"].transpose((1, 0))
#         weight_hh = vars(self)[f"weight_hh_l{l}"].transpose((1, 0))
#         h_out = x_in @ weight_ih + h_in @ weight_hh

#         # Apply the bias
#         if self.bias:
#             bias_ih = vars(self)[f"bias_ih_l{l}"]
#             bias_hh = vars(self)[f"bias_hh_l{l}"]
#             h_out = h_out + bias_ih + bias_hh

#         # Apply the activation function
#         h_out = self.activation(h_out)

#         # Apply dropout
#         if l < self.num_layers - 1 and self.dropout is not None and self.training:
#             h_out = self.dropout(h_out)
#         return h_out
