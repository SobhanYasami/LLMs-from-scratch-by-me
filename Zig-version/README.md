# Zig version: tokenizers from scratch

Same pipeline and output as `../python-version`; read that README for the concepts.

## Layout

```
Zig-version/
├── build.zig
├── README.md
└── src/
    ├── main.zig                 # demo runner (same demos as python main.py)
    ├── ch03.zig                 # chapter 3 demos (attention)
    ├── attention/               # chapter 3: matrix, simple, self_attention, causal, multihead
    └── tokenization/
        ├── root.zig             # public API of the module
        ├── preprocess.zig       # step 1 (book 2.2): splitText, detokenize
        ├── vocab.zig            # step 2 (2.3): Vocab.build
        ├── simple.zig           # step 3 (2.3-2.4): SimpleTokenizer V1 and V2
        ├── dataset.zig          # step 4 (2.6): GptDataset (sliding window)
        ├── dataloader.zig       # step 4b: DataLoader (batch, shuffle, drop last)
        ├── embedding.zig        # step 5 (2.7-2.8): Embedding, InputEmbedding
        ├── rng.zig              # seeded splitmix64 RNG
        └── bpe.zig              # bonus (2.5): GPT-2 byte pair encoder
```

## Running it

From `Zig-version/`:

```bash
zig build run     # the demo
zig build test    # unit tests in src/
```

Written against a Zig 0.15/0.16-dev toolchain (the `std.ArrayList` unmanaged API and `std.fs.File.stdout().writer(...)`). Zig's standard library changes between releases, so a different version may need small edits.

## Chapter 3: attention

`zig build run -- ch03` runs the attention demos (`zig build run` runs every chapter). `src/attention/` has `matrix`, `simple`, `self_attention`, `causal` and `multihead`; the demo is `src/ch03.zig`. Output is identical to Python's.

## Python <-> Zig notes

- No regex in `std`: `splitText` scans bytes. Tokens are slices into the input text, nothing is copied (so keep the text alive).
- Every function that allocates takes an allocator; `main` uses an arena so the demos need no `defer free`.
- Errors are error unions (`error.UnknownToken`, `error.UnknownId`), handled with `try`.
- Tensors are flat `[]f32`, indexed `(b * T + t) * D + d`.
- No tiktoken: `bpe.zig` is a small from-scratch encoder that reads `../assets/gpt2-ranks.txt`. Its IDs match tiktoken's `gpt2`.
- Embedding weights come from a seeded splitmix64, so numbers differ from torch; shapes match.
