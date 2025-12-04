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
from typing import Optional

from ..core import Tensor
from ..nn import Dropout, LayerNorm, Linear, Module
from .container import Sequential


class MultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, device=device)

        self.dropout_layer = Dropout(dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        batch_size = query.shape[0]

        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn_weights = (q @ k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Attention output
        attn_output = attn_weights @ v

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, device=device
        )

        self.linear1 = Linear(d_model, dim_feedforward, device=device)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, device=device)

        self.norm1 = LayerNorm(d_model, device=device)
        self.norm2 = LayerNorm(d_model, device=device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = self.gelu

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, mask=mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    @staticmethod
    def gelu(x: Tensor) -> Tensor:
        return 0.5 * x * (1 + (x * 0.7978845608 * (1 + 0.044715 * x * x)).tanh())


class Transformer(Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layers.append(
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    device=device,
                )
            )

        self.encoder = Sequential(encoder_layers)
        self.d_model = d_model

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # Initialize weights with scaled normal distribution
                p.data.normal_(mean=0.0, std=0.02)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.encoder(src)
