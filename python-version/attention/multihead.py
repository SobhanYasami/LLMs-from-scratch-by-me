"""Step 9 (book 3.6) - multi-head attention.

One attention head learns one way of relating tokens. Several heads, each
with its own smaller query/key/value, can look for different patterns in
parallel (one for grammar, one for topic, ...). Their outputs are joined
and mixed by a final linear layer.

    d_out = 4, num_heads = 2  ->  each head works on head_dim = 2 columns

    W_query columns: [ head 0 | head 1 ]     (same for W_key and W_value)
    per head: causal attention on its own slice -> [T, head_dim]
    concatenate heads -> [T, d_out] -> out_proj

Tip: one big weight matrix cut into column slices is the same as separate
small matrices per head, just faster (one matmul instead of many).
"""

import math

import torch
from torch import nn

from .rng import Rng


class MultiHeadAttention(nn.Module):
    def __init__(self, w_query, w_key, w_value, out_weight, out_bias, num_heads: int, context_length: int):
        super().__init__()
        d_out = w_query.shape[1]
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query, self.W_key, self.W_value = (nn.Parameter(w) for w in (w_query, w_key, w_value))
        self.out_weight = nn.Parameter(out_weight)
        self.out_bias = nn.Parameter(out_bias)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    @classmethod
    def random(cls, d_in: int, d_out: int, num_heads: int, context_length: int, rng: Rng) -> "MultiHeadAttention":
        """Uniform [0, 1) weights (query, key, value, out_proj); zero bias."""
        wq, wk, wv = rng.rand(d_in, d_out), rng.rand(d_in, d_out), rng.rand(d_in, d_out)
        return cls(wq, wk, wv, rng.rand(d_out, d_out), torch.zeros(d_out), num_heads, context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is [batch, T, d_in]; returns [batch, T, d_out]."""
        b, t, _ = x.shape
        # [b, T, d_out] -> [b, T, heads, head_dim] -> [b, heads, T, head_dim]
        def split(m):
            return (x @ m).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.W_query), split(self.W_key), split(self.W_value)
        scores = (q @ k.transpose(-2, -1)).masked_fill(self.mask[:t, :t], -math.inf)
        weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(b, t, -1)  # glue the heads back together
        return context @ self.out_weight + self.out_bias
