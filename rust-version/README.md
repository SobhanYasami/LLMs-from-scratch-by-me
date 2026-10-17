# Rust version: tokenizers from scratch

Same pipeline and output as `../python-version`; read that README for the concepts.

## Layout

```
rust-version/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs              # demo runner (same demos as python main.py)
    ├── lib.rs               # exposes the tokenization module
    └── tokenization/
        ├── mod.rs           # public API of the module
        ├── preprocess.rs    # step 1: split_text, detokenize
        ├── vocab.rs         # step 2: build_vocab
        ├── simple.rs        # step 3: SimpleTokenizerV1 and V2
        ├── dataset.rs       # step 4: GptDataset (sliding window)
        ├── dataloader.rs    # step 4: DataLoader (batch, shuffle, drop last)
        ├── embedding.rs     # step 5: Embedding, InputEmbedding
        └── rng.rs           # seeded splitmix64 RNG (no `rand` dependency)
```

## Running it

From `rust-version/`:

```bash
cargo run
cargo test        # runs the split_text doctest
```

## Chapter 3: attention

`cargo run -- ch03` runs the attention demos (`cargo run` runs every chapter). `src/attention/` has `matrix`, `simple`, `self_attention`, `causal` and `multihead`; the demo is `src/ch03.rs`. The random generator is shared with chapter 2 (`tokenization::rng`). Output is identical to Python's.


## Chapter 4: the GPT model

`cargo run -- ch04` runs the chapter 4 demos (`cargo run --release -- ch04-124m` builds the real 124M model). `src/gpt/` has `layers` (layer norm, GELU, Linear, feed forward), `shortcut` (the 4.4 gradient demo with a hand-written backward pass), `block` (reuses chapter 3's `MultiHeadAttention`) and `model` (`Config`, `GptModel`, `parameter_count`, `generate_text_simple`); the demo is `src/ch04.rs`. `cargo test` checks the book's 163,009,536 parameters. Output is identical to Python's.

## Python <-> Rust notes

- The `regex` crate has no capturing split; `split_text` walks `find_iter` matches by hand.
- Errors are `Result<_, String>` instead of exceptions.
- V2 wraps V1 (composition) instead of subclassing.
- BPE uses `tiktoken-rs`; `r50k_base` is the GPT-2 vocabulary.
- Embedding weights come from a small built-in seeded RNG, so numbers differ from torch; shapes match. Tensors are nested `Vec`s.
- No tensors: the dataset holds `Vec<u32>` windows and `batches` replaces the DataLoader.
