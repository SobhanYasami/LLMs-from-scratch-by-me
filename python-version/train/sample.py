"""Step 19 (book 5.3) - decoding strategies: temperature and top-k sampling.

Greedy decoding always takes the most likely token, so the same prompt gives
the same text, often copied straight from the training data. Two knobs add variety:

- temperature: divide the logits by T before softmax. T < 1 sharpens the
  distribution (more confident), T > 1 flattens it (more surprising).
- top-k: keep only the k best logits, set the others to -inf (probability 0).

Then draw the next token IN PROPORTION to its probability. The draw uses the
repo's shared generator (not torch.multinomial), so every language agrees.
"""

import math

import torch

from attention import Rng


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits (ties at the threshold are kept), -inf for the rest."""
    threshold = torch.topk(logits, k).values[..., -1:]
    return torch.where(logits < threshold, torch.tensor(-math.inf), logits)


def sample_index(probas: torch.Tensor, rng: Rng) -> int:
    """Draw one index in proportion to ``probas`` (a 1-D tensor): inverse-CDF sampling.

    One uniform number u in [0, 1); walk the cumulative sum and stop at the first
    entry where it passes u. Tokens with probability 0 can never be picked.
    """
    u = rng.next_f64()
    cumulative, last = 0.0, 0
    for i, p in enumerate(probas.tolist()):
        if p > 0:
            last = i
        cumulative += p
        if u < cumulative:
            return i
    return last  # rounding left the total a hair below u


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, context_size: int, rng: Rng, temperature: float = 0.0, top_k: int | None = None) -> torch.Tensor:
    """Like ``generate_text_simple`` but with temperature and top-k. ``idx`` is [1, tokens]."""
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]  # scores for the next token: [1, vocab]
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        if temperature > 0.0:
            probas = softmax_with_temperature(logits, temperature)
            idx_next = torch.tensor([[sample_index(probas[0], rng)]])
        else:  # temperature 0 means "no randomness": greedy
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
