"""Educational tokenizer building blocks.

Modules, in the order you learn them:

- ``preprocess``: split raw text into tokens and join them back.
- ``vocab``: map each unique token to an integer ID.
- ``simple``: ``SimpleTokenizerV1`` and ``SimpleTokenizerV2`` built on the above.
- ``dataset``: sliding-window (input, target) pairs for next-token prediction.
- ``embedding``: token + positional embeddings.
"""

from .dataset import GPTDatasetV1, create_dataloader_v1
from .embedding import InputEmbedding
from .preprocess import END_OF_TEXT, UNK_TOKEN, detokenize, split_text
from .simple import SimpleTokenizerV1, SimpleTokenizerV2
from .vocab import build_vocab

__all__ = [
    "InputEmbedding",
    "GPTDatasetV1",
    "create_dataloader_v1",
    "END_OF_TEXT",
    "UNK_TOKEN",
    "SimpleTokenizerV1",
    "SimpleTokenizerV2",
    "build_vocab",
    "detokenize",
    "split_text",
]
