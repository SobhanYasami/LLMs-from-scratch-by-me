# Build an LLM from scratch

Learning repo and future reference, following *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The same material is implemented in three languages with one layout, so each can be compared side by side.

| Dir | Run |
|---|---|
| `python-version/` | `python-version/.venv/bin/python python-version/main.py` |
| `go-version/` | `cd go-version && go run ./cmd` |
| `rust-version/` | `cd rust-version && cargo run` |

All three print identical token IDs. Shared data lives in `assets/`.

## Layout convention

Every version has a README, a demo runner, and a `tokenization` module with one file per step:

1. `preprocess`: split text into tokens
2. `vocab`: token -> ID
3. `simple`: SimpleTokenizerV1 / V2
4. `dataset`: sliding-window (input, target) pairs, batching (DataLoader)
5. `embedding`: token + positional embeddings

New chapters (embeddings, attention, ...) get a new sibling module in each language, in the same order as the book. `python-version/README.md` holds the concept walkthrough; the others note language differences only.

## Docs site

Tutorial and reference live in `docs/` (GitHub Pages): chapter 1 (concepts) and chapter 2 (code in three languages), following the book's section numbers. Code on the site is read from the source files, so regenerate after changing code:

```bash
python3 docs/build.py
```

To publish: GitHub repo -> Settings -> Pages -> Source: "Deploy from a branch", branch `main`, folder `/docs`.
