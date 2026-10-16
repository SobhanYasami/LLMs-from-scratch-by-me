"""Step 6 (book 3.3) - simplified self-attention, no trainable weights.

For every token, build a *context vector*: an enriched embedding that mixes
in information from all the other tokens, weighted by how relevant each is.

    scores  = X @ X.T                  how similar is token i to token j?   [T, T]
    weights = softmax(scores, rows)    turn each row into weights summing to 1
    context = weights @ X              weighted sum of all token vectors   [T, d]

Tip: a dot product measures similarity. Similar vectors -> big score -> big weight.
There is nothing to learn here yet; trainable weights arrive in 3.4.
"""

import torch


def simple_attention(inputs: torch.Tensor):
    """``inputs`` is [T, d]. Returns (scores, weights, context)."""
    scores = inputs @ inputs.T
    weights = torch.softmax(scores, dim=-1)
    context = weights @ inputs
    return scores, weights, context
