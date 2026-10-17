"""Step 17 (book 5.1) - how wrong is the model? Cross entropy loss and perplexity.

For every position the model gives a score (logit) for each of the 50,257
tokens. Softmax turns the scores into probabilities. The loss looks only at
the probability given to the token that REALLY came next:

    probabilities -> pick the target's probability -> log -> average -> minus

    loss = - mean( log p(target) )        (cross entropy)
    perplexity = exp(loss)                "among how many tokens is the model torn?"

A model that guesses uniformly has loss ln(50257) = 10.82. A perfect model has loss 0.
"""

import math

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``logits`` [batch, tokens, vocab], ``targets`` [batch, tokens] -> a single number."""
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())


def perplexity(loss: float) -> float:
    return math.exp(loss)


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model) -> torch.Tensor:
    return cross_entropy(model(input_batch), target_batch)


def calc_loss_loader(data_loader, model, num_batches: int | None = None) -> float:
    """Average loss over (at most ``num_batches``) batches of a data loader."""
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total = 0.0
    for i, (x, y) in enumerate(data_loader):
        if i >= num_batches:
            break
        total += calc_loss_batch(x, y, model).item()
    return total / num_batches
