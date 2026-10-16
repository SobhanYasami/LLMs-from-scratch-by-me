"""Step 2 - Build a vocabulary: a mapping from each token to a unique integer ID.

    vocab = {"!": 0, '"': 1, ..., "<|endoftext|>": 1159, "<|unk|>": 1160}

Computers work with numbers, so every token must get an ID before a model
can use it. The vocabulary is the lookup table between the two worlds.

Tip: a Python ``set`` has no fixed order. Sorting before numbering is what makes
the IDs the same on every run.
"""

from .preprocess import END_OF_TEXT, UNK_TOKEN

SPECIAL_TOKENS = [END_OF_TEXT, UNK_TOKEN]


def build_vocab(tokens: list[str]) -> dict[str, int]:
    """Assign IDs to the unique tokens in sorted order, then append special tokens.

    Sorting makes the IDs deterministic: the same corpus always gives the same vocab.
    Special tokens go last so the regular word IDs stay stable.
    """
    all_tokens = sorted(set(tokens))
    all_tokens.extend(SPECIAL_TOKENS)
    return {token: integer for integer, token in enumerate(all_tokens)}
