"""Attention building blocks (book chapter 3).

Files, in the order you learn them:

- ``simple``: simplified self-attention, no trainable weights (3.3).
- ``self_attention``: scaled dot-product attention with W_query/key/value (3.4).
- ``causal``: hide future tokens with a mask, optional dropout (3.5).
- ``multihead``: several heads in parallel plus an output projection (3.6).
- ``rng``: a seeded random generator shared with the other languages.
"""

from .causal import CausalAttention
from .multihead import MultiHeadAttention
from .rng import Rng, dropout
from .self_attention import SelfAttention
from .simple import simple_attention

__all__ = ["CausalAttention", "MultiHeadAttention", "Rng", "SelfAttention", "dropout", "simple_attention"]
