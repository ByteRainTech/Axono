# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple

from ..core import Tensor
from ..nn import BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU
from .container import Sequential


class CNN(Module):
    """A simple Convolutional Neural Network model."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_channels: List[int] = [32, 64],
        device: str = "cpu",
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        # Add convolutional layers
        for out_channels in hidden_channels:
            layers.extend(
                [
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        device=device,
                    ),
                    BatchNorm2d(out_channels, device=device),
                    ReLU(),
                    MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = Sequential(layers)

        # Calculate the size of flattened features
        self.avgpool = None  # Will be initialized in forward

        # Add classifier
        self.classifier = Sequential(
            [
                Dropout(0.5),
                Linear(hidden_channels[-1] * 7 * 7, 512, device=device),
                ReLU(),
                Dropout(0.5),
                Linear(512, num_classes, device=device),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.avgpool is None:
            self.avgpool = x.shape[2] // 7
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class RNN(Module):
    """A simple Recurrent Neural Network model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input gate
        self.wx = Linear(input_size, hidden_size, device=device)
        self.wh = Linear(hidden_size, hidden_size, device=device)

        # Additional layers
        self.additional_layers = []
        for _ in range(num_layers - 1):
            layer = Linear(hidden_size, hidden_size, device=device)
            self.additional_layers.append(layer)

        self.dropout = Dropout(dropout)
        self.activation = ReLU()

    def forward(self, x: Tensor, hidden: Tensor = None) -> Tuple[Tensor, Tensor]:
        if hidden is None:
            hidden = Tensor.zeros(
                (self.num_layers, x.shape[0], self.hidden_size), device=x.device
            )

        outputs = []
        for t in range(x.shape[1]):
            xt = x[:, t, :]
            h = self.wx(xt) + self.wh(hidden[0])
            h = self.activation(h)
            h = self.dropout(h)

            # Process additional layers
            hidden_states = [h]
            for i, layer in enumerate(self.additional_layers):
                h = layer(h) + hidden[i + 1]
                h = self.activation(h)
                h = self.dropout(h)
                hidden_states.append(h)

            hidden = Tensor.stack(hidden_states)
            outputs.append(h)

        return Tensor.stack(outputs, dim=1), hidden


class LSTM(Module):
    """Long Short-Term Memory network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Gates for each layer
        self.layers = []
        layer_input_size = input_size
        for _ in range(num_layers):
            layer = {
                "forget": Linear(
                    layer_input_size + hidden_size, hidden_size, device=device
                ),
                "input": Linear(
                    layer_input_size + hidden_size, hidden_size, device=device
                ),
                "cell": Linear(
                    layer_input_size + hidden_size, hidden_size, device=device
                ),
                "output": Linear(
                    layer_input_size + hidden_size, hidden_size, device=device
                ),
            }
            self.layers.append(layer)
            layer_input_size = hidden_size

        self.dropout = Dropout(dropout)

    def forward(
        self, x: Tensor, hidden: Tuple[Tensor, Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        if hidden is None:
            h = Tensor.zeros(
                (self.num_layers, batch_size, self.hidden_size), device=x.device
            )
            c = Tensor.zeros(
                (self.num_layers, batch_size, self.hidden_size), device=x.device
            )
            hidden = (h, c)

        h, c = hidden
        output_sequence = []

        for t in range(seq_length):
            xt = x[:, t, :]

            for layer in range(self.num_layers):
                if layer > 0:
                    xt = self.dropout(xt)

                layer_h = h[layer]
                layer_c = c[layer]

                # Concatenate input and hidden state
                combined = Tensor.cat([xt, layer_h], dim=1)

                # Gate computations
                forget_gate = self.layers[layer]["forget"](combined).sigmoid()
                input_gate = self.layers[layer]["input"](combined).sigmoid()
                cell_gate = self.layers[layer]["cell"](combined).tanh()
                output_gate = self.layers[layer]["output"](combined).sigmoid()

                # Update cell and hidden state
                layer_c = forget_gate * layer_c + input_gate * cell_gate
                layer_h = output_gate * layer_c.tanh()

                c[layer] = layer_c
                h[layer] = layer_h
                xt = layer_h

            output_sequence.append(h[-1])

        return Tensor.stack(output_sequence, dim=1), (h, c)
