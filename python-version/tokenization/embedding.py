"""Step 5 - Token and positional embeddings.

A token ID is just an integer. An embedding layer is a lookup table with one
trainable vector per ID, so every token becomes a vector the model can learn from:

    ids [8, 4]  --token_embedding-->  [8, 4, D]
    positions 0..3 --pos_embedding--> [4, D]  (broadcast over the batch)
    input_embeddings = token_embeddings + pos_embeddings   -> [8, 4, D]

Position vectors are needed because attention (later chapters) does not know
word order by itself. GPT-style models learn them as absolute positions.
"""

import torch
from torch import nn


class InputEmbedding(nn.Module):
    """Token embedding plus absolute positional embedding."""

    def __init__(self, vocab_size: int, context_length: int, output_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, output_dim)
        self.pos_embedding = nn.Embedding(context_length, output_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids`` has shape [batch, seq_len]; returns [batch, seq_len, output_dim]."""
        token_embeddings = self.token_embedding(ids)
        positions = torch.arange(ids.shape[1])
        return token_embeddings + self.pos_embedding(positions)
