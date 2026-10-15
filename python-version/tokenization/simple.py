"""Step 3 - Simple tokenizers that combine splitting with the vocabulary.

Both classes expose the same two methods:

    encode(text) -> list[int]   text  -> token IDs
    decode(ids)  -> str         IDs   -> text

V1 only knows the words in its vocabulary and raises ``KeyError`` otherwise.
V2 replaces unknown words with ``<|unk|>`` so encoding never fails.
"""

from .preprocess import UNK_TOKEN, detokenize, split_text


def _invert(vocab: dict[str, int]) -> dict[int, str]:
    """Swap keys and values: ID -> token, needed for decoding."""
    return {integer: token for token, integer in vocab.items()}


class SimpleTokenizerV1:
    """Encode and decode with a vocabulary; unknown tokens are an error."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab              # token -> ID
        self.int_to_str = _invert(vocab)     # ID -> token

    def encode(self, text: str) -> list[int]:
        tokens = split_text(text)
        return [self.str_to_int[token] for token in tokens]  # KeyError if unknown

    def decode(self, ids: list[int]) -> str:
        tokens = [self.int_to_str[i] for i in ids]
        return detokenize(tokens)


class SimpleTokenizerV2(SimpleTokenizerV1):
    """Same as V1, but unknown tokens become ``<|unk|>`` instead of raising."""

    def encode(self, text: str) -> list[int]:
        tokens = split_text(text)
        tokens = [
            token if token in self.str_to_int else UNK_TOKEN
            for token in tokens
        ]
        return [self.str_to_int[token] for token in tokens]
