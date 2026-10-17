# C version: tokenizers from scratch

Same pipeline and output as `../python-version`; read that README for the concepts.

## Layout

```
C-version/
├── Makefile
├── README.md
├── main.c                   # demo runner (same demos as python main.py)
├── ch03.[ch]                # chapter 3 demos (attention)
├── attention/               # chapter 3: matrix, simple, self_attention, causal, multihead
├── ch04.[ch]                # chapter 4 demos (GPT model)
├── gpt/                     # chapter 4: layers, shortcut, block, model
└── tokenization/
    ├── preprocess.[ch]      # step 1 (book 2.2): split_text, detokenize
    ├── vocab.[ch]           # step 2 (2.3): build_vocab
    ├── simple.[ch]          # step 3 (2.3-2.4): SimpleTokenizer V1 and V2
    ├── dataset.[ch]         # step 4 (2.6): GPTDataset (sliding window)
    ├── dataloader.[ch]      # step 4b: DataLoader (batch, shuffle, drop last)
    ├── embedding.[ch]       # step 5 (2.7-2.8): Embedding, InputEmbedding
    ├── rng.[ch]             # seeded splitmix64 RNG
    ├── bpe.[ch]             # bonus (2.5): GPT-2 byte pair encoder
    └── util.[ch]            # checked malloc helpers
```

## Running it

From `C-version/`:

```bash
make run
```

Needs a C11 compiler (`cc`/`gcc`/`clang`) and `make`. No other dependencies.

## Chapter 3: attention

`make run ARGS=ch03` runs the attention demos (`make run` runs every chapter). `attention/` has `matrix` (with a small `Arena` allocator so you free once, not per matrix), `simple`, `self_attention`, `causal` and `multihead`; the demo is `ch03.c`. Output is identical to Python's.


## Chapter 4: the GPT model

`make run ARGS=ch04` runs the chapter 4 demos (`make run ARGS=ch04-124m` builds the real 124M model). `gpt/` has `layers.[ch]` (layer norm, GELU, Linear, feed forward), `shortcut.[ch]` (the 4.4 gradient demo with a hand-written backward pass), `block.[ch]` (reuses chapter 3's `MultiHeadAttention`) and `model.[ch]` (config, `GptModel`, `gpt_parameter_count`, `generate_text_simple`); the demo is `ch04.[ch]`. The model's weights live in one `Arena`; each forward pass takes an `arena_mark` and `arena_reset`s afterwards, so temporaries vanish at once. Output is identical to Python's.

## Python <-> C notes

- No regex: `split_text` scans bytes and cuts at whitespace, `--` and punctuation.
- No dictionaries: `vocab.c` has a small open-addressing hash table; `bpe.c` has another keyed by bytes.
- No exceptions: functions that can fail return `NULL` and write a message into `char err[ERR_LEN]`.
- Memory is manual. Every `*_new` / function returning a pointer has a matching `free` / `*_free`.
- No tensors: an embedding batch is one flat `float` array, indexed `(b * T + t) * D + d`.
- No tiktoken: `bpe.c` is a small from-scratch encoder that reads `../assets/gpt2-ranks.txt`. Its IDs match tiktoken's `gpt2`.
- Embedding weights come from a seeded splitmix64, so numbers differ from torch; shapes match.
