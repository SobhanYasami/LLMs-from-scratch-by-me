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
        └── dataset.rs       # step 4: GptDataset (sliding window)
```

## Running it

From `rust-version/`:

```bash
cargo run
cargo test        # runs the split_text doctest
```

## Python <-> Rust notes

- The `regex` crate has no capturing split; `split_text` walks `find_iter` matches by hand.
- Errors are `Result<_, String>` instead of exceptions.
- V2 wraps V1 (composition) instead of subclassing.
- BPE uses `tiktoken-rs`; `r50k_base` is the GPT-2 vocabulary.
- No tensors: the dataset holds `Vec<u32>` windows and `batches` replaces the DataLoader.
