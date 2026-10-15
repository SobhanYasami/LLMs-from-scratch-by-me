"""Educational tokenizer building blocks.

Modules, in the order you learn them:

- ``preprocess``: split raw text into tokens and join them back.
- ``vocab``: map each unique token to an integer ID.
- ``simple``: ``SimpleTokenizerV1`` and ``SimpleTokenizerV2`` built on the above.
"""

from .preprocess import END_OF_TEXT, UNK_TOKEN, detokenize, split_text
from .simple import SimpleTokenizerV1, SimpleTokenizerV2
from .vocab import build_vocab

__all__ = [
    "END_OF_TEXT",
    "UNK_TOKEN",
    "SimpleTokenizerV1",
    "SimpleTokenizerV2",
    "build_vocab",
    "detokenize",
    "split_text",
]
