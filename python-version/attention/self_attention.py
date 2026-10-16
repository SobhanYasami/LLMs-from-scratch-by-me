"""Step 7 (book 3.4) - self-attention with trainable weights (scaled dot-product).

Three weight matrices turn each input vector into three roles:

    query = x @ W_query    "what am I looking for?"
    key   = x @ W_key      "what do I offer to a search?"
    value = x @ W_value    "what do I hand over if I am chosen?"

    scores  = query @ key.T                 how well does each query match each key?
    weights = softmax(scores / sqrt(d_k))   scale, then normalise each row
    context = weights @ value               mix the values

Why divide by sqrt(d_k)? Big dot products push softmax towards a hard 0/1
choice, which makes gradients tiny. Scaling keeps training healthy.

Tip: do not confuse *weight parameters* (W_*, learned) with *attention weights*
(softmax output, computed fresh for every input).
"""

import math

import torch
from torch import nn

from .rng import Rng


class SelfAttention(nn.Module):
    def __init__(self, w_query: torch.Tensor, w_key: torch.Tensor, w_value: torch.Tensor):
        super().__init__()
        self.W_query = nn.Parameter(w_query)
        self.W_key = nn.Parameter(w_key)
        self.W_value = nn.Parameter(w_value)

    @classmethod
    def random(cls, d_in: int, d_out: int, rng: Rng) -> "SelfAttention":
        """Weights drawn uniform in [0, 1), in query, key, value order."""
        return cls(rng.rand(d_in, d_out), rng.rand(d_in, d_out), rng.rand(d_in, d_out))

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        queries, keys = x @ self.W_query, x @ self.W_key
        scores = queries @ keys.T
        return torch.softmax(scores / math.sqrt(keys.shape[-1]), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is [T, d_in]; returns context vectors [T, d_out]."""
        return self.attention_weights(x) @ (x @ self.W_value)
