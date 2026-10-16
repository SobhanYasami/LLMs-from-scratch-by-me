"""Step 1 - Split raw text into tokens (and join them back).

A token here is a word, a punctuation mark, or a whitespace-free piece of text.
The regex keeps the delimiters, so punctuation becomes its own token:

    "Hello, world!"  ->  ['"', 'Hello', ',', 'world', '!', '"']   (roughly)

Note: ``re.split`` with a *capturing* group returns the separators too.
Without the parentheses in ``SPLIT_PATTERN`` they would be dropped.
"""

import re

# Tip: start reading here, then vocab -> simple -> dataset -> embedding.
# Each file uses the one before it.

# Captures: , . ? _ ! " ( ) ' , the double dash "--", and any whitespace.
SPLIT_PATTERN = r'([,.?_!"()\']|--|\s)'

# Used when decoding: removes the space that " ".join() put before punctuation.
DETOKENIZE_PATTERN = r'\s+([,.?!"()\'])'

# Special tokens. They are not produced by splitting text; they are added on purpose.
END_OF_TEXT = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"


def split_text(text: str) -> list[str]:
    """Return the tokens of ``text``, with whitespace-only pieces removed.

    >>> split_text('Hello, world!')
    ['Hello', ',', 'world', '!']
    """
    pieces = re.split(SPLIT_PATTERN, text)
    # re.split leaves empty strings and lone spaces behind; drop them.
    return [piece.strip() for piece in pieces if piece.strip()]


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into text, undoing the spaces added before punctuation.

    >>> detokenize(['Hello', ',', 'world', '!'])
    'Hello, world!'

    This is not a perfect inverse: the original spacing is lost, so
    ``detokenize(split_text(t)) == t`` only holds for "simple" text.
    """
    text = " ".join(tokens)
    return re.sub(DETOKENIZE_PATTERN, r"\1", text)
