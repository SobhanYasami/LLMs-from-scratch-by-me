# Go version: tokenizers from scratch

Same pipeline and output as `../python-version`; read that README for the concepts.

## Layout

```
go-version/
├── cmd/main.go              # demo runner (same demos as python main.py)
├── README.md
└── tokenization/
    ├── preprocess.go        # step 1: SplitText, Detokenize
    ├── vocab.go             # step 2: BuildVocab
    ├── simple.go            # step 3: SimpleTokenizerV1 and V2
    ├── dataset.go           # step 4: GPTDataset (sliding window)
    ├── dataloader.go        # step 4: DataLoader (batch, shuffle, drop last)
    └── embedding.go         # step 5: Embedding, InputEmbedding
```

## Running it

From `go-version/`:

```bash
go run ./cmd
```

## Chapter 3: attention

`go run ./cmd ch03` runs the attention demos (`go run ./cmd` runs every chapter). The `attention/` package has `matrix`, `rng`, `simple`, `self_attention`, `causal` and `multihead`; the demo is `cmd/ch03.go`. Output is identical to Python's.


## Chapter 4: the GPT model

`go run ./cmd ch04` runs the chapter 4 demos (`go run ./cmd ch04-124m` builds the real 124M model). The `gpt/` package has `layers.go` (layer norm, GELU, Linear, feed forward), `shortcut.go` (the 4.4 gradient demo, with a hand-written backward pass instead of autograd), `block.go` (transformer block, reusing chapter 3's `MultiHeadAttention`) and `model.go` (config, `Model`, `ParameterCount`, `GenerateTextSimple`); the demo is `cmd/ch04.go`. `go test ./gpt` checks the book's 163,009,536 parameters. Output is identical to Python's.


## Chapter 5: pretraining

`go run ./cmd ch05` trains the tiny GPT for two epochs, then runs the decoding, checkpoint and loading demos; `go run ./cmd ch05-gpt2` loads OpenAI's GPT-2 weights (download first, see `train/openai.go`). The `train/` package has `loss.go`, `grad.go` (the backward pass written by hand: no autograd in Go) with `adamw.go`, `sample.go`, `checkpoint.go` and `openai.go`; the demo is `cmd/ch05.go`. `go test ./train` includes a finite-difference gradient check. Output is identical to Rust, C and Zig; Python/torch agrees until late in training.

## Python <-> Go notes

- `re.split` with a capturing group has no Go equivalent; `splitKeep` in `preprocess.go` walks the matches by hand.
- Python raises `KeyError`; Go `Encode`/`Decode` return `error`.
- V2 embeds V1 (struct embedding) instead of subclassing.
- BPE uses `tiktoken-go` with the offline loader. Go has no `gpt2` name; `r50k_base` is the same vocabulary.
- Embedding weights are random normal from `math/rand`, so numbers differ from torch; shapes match. Tensors are nested slices.
- No tensors: the dataset holds `[]int` slices and `Batches` replaces the DataLoader.
