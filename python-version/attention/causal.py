"""Step 8 (book 3.5) - causal attention: hide the future.

A GPT writes text left to right, so token i may only look at tokens 0..i.
We hide the later tokens with a mask. Set their scores to -inf *before*
softmax: exp(-inf) = 0, and the remaining weights in each row still sum to 1.

    scores (T x T)                       after the mask
    [ s00 s01 s02 ]                      [ s00 -inf -inf ]
    [ s10 s11 s12 ]      -------->       [ s10  s11 -inf ]
    [ s20 s21 s22 ]                      [ s20  s21  s22 ]

Dropout (training only) randomly zeroes some attention weights so the model
does not lean on any single connection.
"""

import math

import torch
from torch import nn

from .rng import Rng, dropout
from .self_attention import SelfAttention


class CausalAttention(SelfAttention):
    def __init__(self, w_query, w_key, w_value, context_length: int, dropout_p: float = 0.0, rng: Rng | None = None):
        super().__init__(w_query, w_key, w_value)
        self.dropout_p = dropout_p
        self.rng = rng
        # True above the diagonal = "future" positions to hide. A buffer is saved
        # with the model but is not a learned parameter.
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def attention_weights(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """``x`` is [T, d_in] or [batch, T, d_in]."""
        queries, keys = x @ self.W_query, x @ self.W_key
        scores = queries @ keys.transpose(-2, -1)
        t = x.shape[-2]
        scores = scores.masked_fill(self.mask[:t, :t], -math.inf)
        weights = torch.softmax(scores / math.sqrt(keys.shape[-1]), dim=-1)
        if train and self.dropout_p > 0:
            weights = dropout(weights, self.dropout_p, self.rng)
        return weights

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        return self.attention_weights(x, train) @ (x @ self.W_value)
