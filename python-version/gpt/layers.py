"""Steps 10-12 (book 4.2 and 4.3) - layer normalization, GELU and the feed forward block.

Layer normalization rescales each token's vector to mean 0 and variance 1,
which keeps activations in a range where training is stable:

    x  [.., D]  ->  (x - mean) / sqrt(var + eps)  ->  scale * that + shift

The mean and variance are taken over the last axis (the D numbers of ONE
token), never across tokens or across the batch. ``scale`` and ``shift`` are
learned, so the model can undo the normalization where that helps.

GELU is a smooth cousin of ReLU, and the feed forward block expands each
token's vector 4x, applies GELU, and shrinks it back:

    [.., D] -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> [.., D]
"""

import math

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # divide by n, like GPT-2
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """The tanh approximation GPT-2 was trained with."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
