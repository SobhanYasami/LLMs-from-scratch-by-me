# Build an LLM from scratch

Learning repo and future reference, following *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The same material is implemented in five languages with one layout, so each can be compared side by side.

| Dir | Run |
|---|---|
| `python-version/` | `python-version/.venv/bin/python python-version/main.py` |
| `go-version/` | `cd go-version && go run ./cmd` |
| `rust-version/` | `cd rust-version && cargo run` |
| `C-version/` | `cd C-version && make run` |
| `Zig-version/` | `cd Zig-version && zig build run` |

All five print identical token IDs, and identical output for chapters 3 and 4 (the GPT model is built from the same seeded random weights in every language). Chapter 5 training is identical in Go, Rust, C and Zig (byte-identical saved models); Python/torch agrees until late in training. Shared data lives in `assets/` (`the-verdict.txt`, and `gpt2-ranks.txt`, the GPT-2 BPE vocabulary used by the C and Zig encoders).

## Layout convention

Every version has a README, a demo runner, and a `tokenization` module with one file per step:

1. `preprocess`: split text into tokens
2. `vocab`: token -> ID
3. `simple`: SimpleTokenizerV1 / V2
4. `dataset`: sliding-window (input, target) pairs, batching (DataLoader)
5. `embedding`: token + positional embeddings

Chapter 3 adds a sibling `attention` module with `simple`, `self_attention`, `causal` and `multihead`. Chapter 4 adds a `gpt` module with `layers` (layer norm, GELU, feed forward), `shortcut`, `block` and `model` (the GPT model, parameter count, text generation). Each `main` runs one chapter with an argument (`ch02`, `ch03`, `ch04`, `ch05`) or all of them; `ch04-124m` runs the real 124M-parameter GPT-2 (about 1 GB of memory, build in release mode). Chapter 5 adds a `train` module (loss, backward pass, AdamW, sampling, checkpoints, OpenAI weights) and runs with `ch05`; `ch05-gpt2` loads OpenAI's pretrained GPT-2 (download in `site/content/en/ch05-5-openai-weights.html`, about 500 MB).

New chapters (embeddings, attention, ...) get a new sibling module in each language, in the same order as the book. `python-version/README.md` holds the concept walkthrough; the others note language differences only.

## Docs site

A living tutorial and reference for the whole book, built with [Hugo](https://gohugo.io/) from `site/` and published with GitHub Pages. It is built to grow, and it is available in English and Persian:

- `site/content/en/*.html`: one file per page (small front-matter block, then HTML). Add a file to add a page.
- `site/content/fa/*.html`: the Persian translation of each page (same file name).
- `site/data/chapters.yaml`: the whole book (chapters, sections, both languages) and which languages implement each chapter. The roadmap, progress bars, chapter pages and navigation are generated from it, so unwritten parts show up as "planned".
- `site/data/code.yaml`: which source files each code tab shows. Code on the site is read from the real source files at build time and highlighted by Hugo, so it never drifts from the code.
- `site/i18n/`: interface strings per language. `site/layouts/`, `site/assets/`: templates, CSS and JS.

```bash
cd site
hugo server        # live preview at http://localhost:1313/
hugo --minify      # build into site/public/ (not committed)
```

Needs Hugo extended 0.158 or newer. Step-by-step guide for adding pages, extras, chapters, code and languages: the "Contributing" page of the site (`site/content/en/contributing.html`).

To publish: repository Settings -> Pages -> Source: **GitHub Actions**. The workflow in `.github/workflows/hugo.yml` builds and deploys the site on every push to `main`.
