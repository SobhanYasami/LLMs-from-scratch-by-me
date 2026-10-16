"""Chapter 3 demos: attention, one idea at a time (book 3.3 to 3.6).

Run:  python-version/.venv/bin/python python-version/main.py ch03

Tip: read the demos top to bottom; each builds on the one before. The numbers
in 3.3 and 3.4 match the book exactly (3.4 uses the book's seed-123 weights,
hard-coded so every language shares them).
"""

import math

import torch

from attention import CausalAttention, MultiHeadAttention, Rng, SelfAttention, dropout, simple_attention

# "Your journey starts with one step." as 3-dimensional embeddings (book 3.3.1).
INPUTS = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)

# The book's torch.manual_seed(123) weights (d_in=3, d_out=2), written out so no language needs torch.
W_QUERY = [[0.29611194, 0.51656228], [0.25167072, 0.68855679], [0.07397246, 0.86652195]]
W_KEY = [[0.13657987, 0.10247904], [0.18405646, 0.72644675], [0.31525391, 0.68710667]]
W_VALUE = [[0.07563531, 0.19663817], [0.31641197, 0.40174013], [0.11856830, 0.82739538]]


def fmt_row(values) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"


def show(label: str, t: torch.Tensor) -> None:
    t = t.detach()
    if t.dim() == 1:
        print(f"{label}: {fmt_row(t.tolist())}")
    else:
        print(f"{label}:")
        for row in t.tolist():
            print("  " + fmt_row(row))


def demo_simple_attention() -> None:
    """3.3: one query by hand, then all tokens at once."""
    print("== 3.3.1 simple attention for one query (x2, 'journey') ==")
    query = INPUTS[1]
    scores_2 = INPUTS @ query  # dot product of every token with the query
    weights_2 = torch.softmax(scores_2, dim=0)
    context_2 = weights_2 @ INPUTS
    show("scores    ", scores_2)
    show("weights   ", weights_2)
    show("context z2", context_2)

    print("== 3.3.2 all tokens at once ==")
    scores, weights, context = simple_attention(INPUTS)
    show("scores (T x T)", scores)
    show("weights (rows sum to 1)", weights)
    show("row sums", weights.sum(dim=1))
    show("context vectors (T x d)", context)


def demo_self_attention() -> None:
    """3.4: query/key/value with trainable (here: fixed) weights."""
    print("== 3.4 self-attention with trainable weights (d_in=3, d_out=2) ==")
    wq, wk, wv = (torch.tensor(w) for w in (W_QUERY, W_KEY, W_VALUE))
    sa = SelfAttention(wq, wk, wv)
    x2 = INPUTS[1]
    query_2, keys, values = x2 @ wq, INPUTS @ wk, INPUTS @ wv
    show("query_2", query_2)
    print(f"keys shape: {tuple(keys.shape)}, values shape: {tuple(values.shape)}")
    scores_2 = query_2 @ keys.T
    show("scores_2 (q2 . keys)", scores_2)
    weights_2 = torch.softmax(scores_2 / math.sqrt(keys.shape[-1]), dim=-1)
    show("weights_2 (scaled)", weights_2)
    show("context_2", weights_2 @ values)
    show("all context vectors (T x d_out)", sa(INPUTS))


def demo_causal_attention() -> None:
    """3.5: mask the future, then dropout."""
    print("== 3.5 causal attention ==")
    wq, wk, wv = (torch.tensor(w) for w in (W_QUERY, W_KEY, W_VALUE))
    ca = CausalAttention(wq, wk, wv, context_length=6)
    show("causal weights (upper triangle is 0)", ca.attention_weights(INPUTS))
    show("causal context vectors", ca(INPUTS))
    dropped = dropout(torch.ones(6, 6), 0.5, Rng(123))
    show("dropout 0.5 on ones(6, 6), seed 123 (kept values are doubled)", dropped)


def demo_multihead() -> None:
    """3.6: split the work across heads."""
    print("== 3.6 multi-head attention (d_in=3, d_out=2, heads=2) ==")
    mha = MultiHeadAttention.random(d_in=3, d_out=2, num_heads=2, context_length=6, rng=Rng(123))
    batch = torch.stack((INPUTS, INPUTS), dim=0)  # two copies = a batch of 2
    out = mha(batch)
    print(f"input shape: {tuple(batch.shape)}  output shape: {tuple(out.shape)}")
    show("output[0]", out[0])


def run() -> None:
    demo_simple_attention()
    demo_self_attention()
    demo_causal_attention()
    demo_multihead()
