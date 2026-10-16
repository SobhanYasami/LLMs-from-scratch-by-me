"""Demo runner for the educational tokenizers.

Run from anywhere:

    python-version/.venv/bin/python python-version/main.py

Pipeline:
    raw text -> split_text -> build_vocab -> SimpleTokenizerV1/V2 -> tiktoken (GPT-2 BPE)

See python-version/README.md for a walkthrough of each step.
"""

from importlib.metadata import version
from pathlib import Path

import tiktoken

from tokenization import (
    END_OF_TEXT,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    split_text,
)

# Resolved from this file, so the script works no matter the current directory.
DATA_PATH = Path(__file__).resolve().parent.parent / "assets" / "the-verdict.txt"

V1_TEXT = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
V2_TEXT = f"Hello, do you like tea? {END_OF_TEXT} In the sunlit terraces of the palace."

def load_textflie():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text


def load_vocab() -> dict[str, int]:
    """Read the corpus, tokenize it, and build the vocabulary."""
    raw_text = DATA_PATH.read_text(encoding="utf-8")
    return build_vocab(split_text(raw_text))


def show_vocab_tail(vocab: dict[str, int], count: int = 5) -> None:
    """Print the last ``count`` vocabulary entries (the highest IDs)."""
    for item in list(vocab.items())[-count:]:
        print(item)


def demo_v1(vocab: dict[str, int]) -> None:
    """V1 works only on words it has seen in the corpus."""
    tokenizer = SimpleTokenizerV1(vocab)
    print("version 1 tokenizer :")
    ids = tokenizer.encode(V1_TEXT)
    print(ids)
    print(tokenizer.decode(ids))
    print("-------------------------")


def demo_v2(vocab: dict[str, int]) -> None:
    """V2 maps unknown words to <|unk|> and handles the <|endoftext|> separator."""
    tokenizer = SimpleTokenizerV2(vocab)
    print("version 2 tokenizer :")
    ids = tokenizer.encode(V2_TEXT)
    print(ids)
    print(tokenizer.decode(ids))
    print("---------------------")


def demo_bpe() -> None:
    """Byte pair encoding with OpenAI's tiktoken (GPT-2 vocabulary, ~50k tokens).

    Unlike the simple tokenizers above, BPE learns sub-word pieces, so it never
    needs an <|unk|> token: any string can be broken into known pieces.
    """
    raw_text = load_textflie()
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(V2_TEXT, allowed_special={END_OF_TEXT})
    enc_text = tokenizer.encode(raw_text)
    
    """we remove the first 50 tokens from the dataset for demonstration
    purposes as it results in a slightly more interesting text passage in the next
    steps"""
    enc_sample = enc_text[50:]
    
    """One of the easiest and most intuitive ways to create the input-target pairs for
    the next-word prediction task is to create two variables, x and y, where x
    contains the input tokens and y contains the targets, which are the inputs
    shifted by 1:"""
    context_size = 4 #A
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:      {y}")
    
    """Processing the inputs along with the targets, which are the inputs shifted by
    one position, we can then create the next-word predictio"""
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)
        
    """For illustration purposes, let's repeat the previous code but convert the token
    IDs into text:"""
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    

def main() -> None:
    vocab = load_vocab()
    # show_vocab_tail(vocab)
    # demo_v1(vocab)
    # demo_v2(vocab)
    demo_bpe()


if __name__ == "__main__":
    main()
