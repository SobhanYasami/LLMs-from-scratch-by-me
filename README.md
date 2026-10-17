# Build an LLM from scratch

Learning repo and future reference, following *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The same material is implemented in five languages with one layout, so each can be compared side by side.

| Dir | Run |
|---|---|
| `python-version/` | `python-version/.venv/bin/python python-version/main.py` |
| `go-version/` | `cd go-version && go run ./cmd` |
| `rust-version/` | `cd rust-version && cargo run` |
| `C-version/` | `cd C-version && make run` |
| `Zig-version/` | `cd Zig-version && zig build run` |

All five print identical token IDs. Shared data lives in `assets/` (`the-verdict.txt`, and `gpt2-ranks.txt`, the GPT-2 BPE vocabulary used by the C and Zig encoders).

## Layout convention

Every version has a README, a demo runner, and a `tokenization` module with one file per step:

1. `preprocess`: split text into tokens
2. `vocab`: token -> ID
3. `simple`: SimpleTokenizerV1 / V2
4. `dataset`: sliding-window (input, target) pairs, batching (DataLoader)
5. `embedding`: token + positional embeddings

Chapter 3 adds a sibling `attention` module with `simple`, `self_attention`, `causal` and `multihead`. Each `main` runs one chapter with an argument (`ch02`, `ch03`) or all of them.

New chapters (embeddings, attention, ...) get a new sibling module in each language, in the same order as the book. `python-version/README.md` holds the concept walkthrough; the others note language differences only.

## Docs site

A living tutorial and reference for the whole book, published with GitHub Pages from `docs/`. It is built to grow:

- `docs/pages/*.html`: one file per page (small front-matter block, then HTML). Add a file to add a page.
- `docs/pages/fa/*.html`: Persian translation of each page (same file name). Untranslated pages fall back to English. UI strings and Persian chapter names are in `docs/i18n.py`. The Persian site is built to `docs/fa/`.
- `docs/syllabus.py`: the whole book (chapters, sections), the languages, and which source files each code tab shows. The roadmap, progress bars, chapter pages and navigation are generated from it, so unwritten parts show up as "planned".
- Code on the site is read from the source files, so regenerate after any change and commit the result:

```bash
python3 docs/build.py
```

Step-by-step guide for adding pages, extras, chapters, code and languages: `docs/pages/contributing.html` (published as "Contributing").

To publish: GitHub repo -> Settings -> Pages -> Source: "Deploy from a branch", branch `main`, folder `/docs`.
