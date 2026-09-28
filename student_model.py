"""Student GPT variant with a parameter-matched SwiGLU feed-forward network.

The attention, positional embeddings, residual layout, initialization and tied
output embedding intentionally match the supplied baseline.  The controlled
change is the token-wise feed-forward sublayer: GELU MLP -> SwiGLU MLP.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Gated MLP: output(SiLU(gate(x)) * value(x))."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class SwiGLUBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_hidden: int, dropout: float = 0.0):
        super().__init__()
        if width % heads != 0:
            raise ValueError('width must be divisible by heads')
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = SwiGLU(width, mlp_hidden)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class SwiGLUGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        # A standard 4*width GELU MLP uses about 8*width^2 matrix weights.
        # SwiGLU has three matrices, so hidden ~= 8*width/3 matches capacity.
        mlp_hidden = config.get('mlp_hidden', round(8 * width / 3))
        dropout = config.get('dropout', 0.0)
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SwiGLUBlock(width, config['heads'], mlp_hidden, dropout)
            for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding_dropout(self.token(ids) + self.pos(positions))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return SwiGLUGPT(config)
