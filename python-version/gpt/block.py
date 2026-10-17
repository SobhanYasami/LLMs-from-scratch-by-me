"""Step 14 (book 4.5) - the transformer block.

One block is two sub-layers, each wrapped in the same recipe
(normalize first, run the layer, drop out, add the input back):

    x = x + dropout(attention(norm1(x)))      <- tokens look at each other
    x = x + dropout(feed_forward(norm2(x)))   <- each token is processed alone

The output has the same shape as the input, so blocks can be stacked as many
times as you like. Normalizing BEFORE each layer is called pre-LayerNorm; it
trains better than the original transformer's normalize-after.

Tip: the attention here is the chapter 3 ``MultiHeadAttention`` unchanged.
"""

import torch
from torch import nn

from attention import MultiHeadAttention

from .layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["emb_dim"]
        zeros = lambda *shape: torch.zeros(*shape)  # noqa: E731 - real values come from init_weights
        self.att = MultiHeadAttention(zeros(d, d), zeros(d, d), zeros(d, d), zeros(d, d), zeros(d), cfg["n_heads"], cfg["context_length"],
                                  qkv_bias=(zeros(d), zeros(d), zeros(d)) if cfg["qkv_bias"] else None)
        self.ff = FeedForward(d)
        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # active only in .train() mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # add the original input back

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut
