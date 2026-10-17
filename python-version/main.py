"""Demo runner for the educational tokenizers and data sampling.

Run from anywhere:

    python-version/.venv/bin/python python-version/main.py          # all chapters
    python-version/.venv/bin/python python-version/main.py ch03     # one chapter

Pipeline:
    raw text -> split_text -> build_vocab -> SimpleTokenizerV1/V2 -> tiktoken (GPT-2 BPE)
    -> sliding-window dataset

Tip: read ``main()`` first, then jump into ``tokenization/`` following each demo.
See python-version/README.md for a walkthrough of each step.
"""

import sys
from pathlib import Path

import tiktoken
import torch

from tokenization import (
    END_OF_TEXT,
    GPTDatasetV1,
    InputEmbedding,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    create_dataloader_v1,
    split_text,
)

# Resolved from this file, so the script works no matter the current directory.
DATA_PATH = Path(__file__).resolve().parent.parent / "assets" / "the-verdict.txt"

V1_TEXT = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
V2_TEXT = f"Hello, do you like tea? {END_OF_TEXT} In the sunlit terraces of the palace."


def load_text() -> str:
    return DATA_PATH.read_text(encoding="utf-8")


def load_vocab() -> dict[str, int]:
    """Read the corpus, tokenize it, and build the vocabulary."""
    return build_vocab(split_text(load_text()))


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
    tokenizer = tiktoken.get_encoding("gpt2")
    print("bpe tokenizer :")
    ids = tokenizer.encode(V2_TEXT, allowed_special={END_OF_TEXT})
    print(ids)
    print(tokenizer.decode(ids))
    print("---------------------")


def demo_sampling() -> None:
    """Next-word prediction pairs: the target is the input shifted by one."""
    tokenizer = tiktoken.get_encoding("gpt2")
    # Drop the first 50 tokens: gives a slightly more interesting passage.
    enc_sample = tokenizer.encode(load_text())[50:]

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1 : context_size + 1]
    print("data sampling :")
    print(f"x: {x}")
    print(f"y:      {y}")

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

    # The same idea, packaged: every window of the whole text, in batches.
    dataset = GPTDatasetV1(load_text(), tokenizer, max_length=4, stride=4)
    print(f"windows: {len(dataset)}, first pair: {dataset[0]}")
    loader = create_dataloader_v1(load_text(), tokenizer, batch_size=8, max_length=4, stride=4, shuffle=False)
    inputs, targets = next(iter(loader))
    print("inputs:\n", inputs)
    print("targets:\n", targets)
    print("---------------------")


def demo_dataloader() -> None:
    """Batches of windows. stride < max_length overlaps windows; stride == max_length does not."""
    tokenizer = tiktoken.get_encoding("gpt2")
    text = load_text()
    print("data loader :")
    loader = create_dataloader_v1(text, tokenizer, batch_size=1, max_length=4, stride=1, shuffle=False)
    it = iter(loader)
    print(next(it))  # first window
    print(next(it))  # second window: shifted by one token
    loader = create_dataloader_v1(text, tokenizer, batch_size=8, max_length=4, stride=4, shuffle=False)
    inputs, targets = next(iter(loader))
    print("inputs shape:", tuple(inputs.shape))
    print("---------------------")


def demo_embeddings() -> None:
    """Token IDs -> vectors, then add position vectors."""
    torch.manual_seed(123)  # embedding weights start random; fix them for repeatable output
    tokenizer = tiktoken.get_encoding("gpt2")
    max_length, output_dim = 4, 256
    loader = create_dataloader_v1(
        load_text(), tokenizer, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    inputs, _ = next(iter(loader))

    embedding = InputEmbedding(tokenizer.n_vocab, max_length, output_dim)
    token_embeddings = embedding.token_embedding(inputs)
    pos_embeddings = embedding.pos_embedding(torch.arange(max_length))
    input_embeddings = embedding(inputs)
    print("embeddings :")
    # Shapes are the thing to notice: [batch, tokens] grows a new axis of size D.
    print("token ids shape:     ", tuple(inputs.shape))
    print("token embeddings:    ", tuple(token_embeddings.shape))
    print("position embeddings: ", tuple(pos_embeddings.shape), " <- one vector per position, shared by all rows")
    print("input embeddings:    ", tuple(input_embeddings.shape), " <- token + position")
    print("---------------------")


def run_chapter2() -> None:
    vocab = load_vocab()
    show_vocab_tail(vocab)
    demo_v1(vocab)
    demo_v2(vocab)
    demo_bpe()
    demo_sampling()
    demo_dataloader()
    demo_embeddings()


def main() -> None:
    """``main.py [ch02|ch03|ch04|ch04-124m]``: run one chapter's demos, or all three when no argument is given."""
    chapters = sys.argv[1:] or ["ch02", "ch03", "ch04"]
    for chapter in chapters:
        if chapter == "ch02":
            run_chapter2()
        elif chapter == "ch03":
            import ch03

            ch03.run()
        elif chapter == "ch04":
            import ch04

            ch04.run()
        elif chapter == "ch04-124m":
            import ch04

            ch04.run_124m()
        else:
            sys.exit(f"unknown chapter {chapter!r}; use ch02, ch03, ch04 or ch04-124m")


if __name__ == "__main__":
    main()
