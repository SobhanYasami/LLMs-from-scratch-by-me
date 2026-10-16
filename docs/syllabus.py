"""The map of the whole book, and which language implements which part.

This file is the single place to update when the project grows:

- CHAPTERS: every chapter and section of the book. A section is "ready" once a
  page in docs/pages/ lists it in its `section:` front matter. Sections nobody
  has written yet show up as "planned" on the roadmap automatically.
- LANGS: the implementations. Add a language here and give it entries in CODE.
- CODE: for each code key used by a page's `code:` front matter, the source
  files to show in each language tab (paths from the repo root).

See docs/pages/contributing.html for the step-by-step guide.
"""

# (key, label, directory). Order = tab order.
LANGS = [
    ("python", "Python", "python-version"),
    ("go", "Go", "go-version"),
    ("rust", "Rust", "rust-version"),
    ("c", "C", "C-version"),
    ("zig", "Zig", "Zig-version"),
]
HLJS_LANG = {"python": "python", "go": "go", "rust": "rust", "c": "c", "zig": "zig"}

# status of a chapter's code: which languages implement its sections
ALL = [k for k, _, _ in LANGS]

CHAPTERS = [
    dict(num=1, title="Understanding large language models", code=False,
         summary="What an LLM is, how it is trained, and what a GPT looks like. No code.",
         sections=[
             ("1.1", "What is an LLM?"), ("1.2", "Applications of LLMs"),
             ("1.3", "Stages of building and using LLMs"), ("1.4", "Using LLMs for different tasks"),
             ("1.5", "Utilizing large datasets"), ("1.6", "A closer look at the GPT architecture"),
             ("1.7", "Building a large language model"),
         ]),
    dict(num=2, title="Working with text data", code=True, langs=ALL,
         summary="Turn raw text into model input: tokens, IDs, byte pair encoding, training pairs and embeddings.",
         sections=[
             ("2.1", "Understanding word embeddings"), ("2.2", "Tokenizing text"),
             ("2.3", "Converting tokens into token IDs"), ("2.4", "Adding special context tokens"),
             ("2.5", "Byte pair encoding"), ("2.6", "Data sampling with a sliding window"),
             ("2.7", "Creating token embeddings"), ("2.8", "Encoding word positions"),
         ]),
    dict(num=3, title="Coding attention mechanisms", code=True, langs=ALL,
         summary="Why attention exists, then self-attention, causal masking, dropout and multi-head attention.",
         sections=[
             ("3.1", "The problem with modeling long sequences"),
             ("3.2", "Capturing data dependencies with attention mechanisms"),
             ("3.3", "Attending to different parts of the input with self-attention"),
             ("3.4", "Implementing self-attention with trainable weights"),
             ("3.5", "Hiding future words with causal attention"),
             ("3.6", "Extending single-head attention to multi-head attention"),
         ]),
    dict(num=4, title="Implementing a GPT model from scratch", code=True, langs=[],
         summary="Layer normalization, feed-forward networks, shortcut connections, the transformer block, the full GPT model and text generation.",
         sections=[
             ("4.1", "Coding an LLM architecture"), ("4.2", "Normalizing activations with layer normalization"),
             ("4.3", "Implementing a feed forward network with GELU activations"),
             ("4.4", "Adding shortcut connections"),
             ("4.5", "Connecting attention and linear layers in a transformer block"),
             ("4.6", "Coding the GPT model"), ("4.7", "Generating text"),
         ]),
    dict(num=5, title="Pretraining on unlabeled data", code=True, langs=[],
         summary="Loss, the training loop, decoding strategies (temperature, top-k), saving weights and loading OpenAI's pretrained weights.",
         sections=[
             ("5.1", "Evaluating generative text models"), ("5.2", "Training an LLM"),
             ("5.3", "Decoding strategies to control randomness"),
             ("5.4", "Loading and saving model weights in PyTorch"),
             ("5.5", "Loading pretrained weights from OpenAI"),
         ]),
    dict(num=6, title="Finetuning for text classification", code=True, langs=[],
         summary="Adapt the pretrained GPT into a spam classifier: prepare a labelled dataset, add a classification head, finetune and evaluate.",
         sections=[]),
    dict(num=7, title="Finetuning to follow instructions", code=True, langs=[],
         summary="Instruction datasets, batching with padding, finetuning, and evaluating the answers.",
         sections=[]),
    dict(num="A", title="Companion material", code=False,
         summary="A: PyTorch primer. B: references. C: exercise solutions. D: training-loop extras (warmup, cosine decay, clipping). E: LoRA finetuning.",
         sections=[
             ("A", "Introduction to PyTorch"), ("B", "References and further reading"),
             ("C", "Exercise solutions"), ("D", "Adding bells and whistles to the training loop"),
             ("E", "Parameter-efficient finetuning with LoRA"),
         ]),
]

# Source files per code key, relative to each language's directory.
CODE = {
    "attn_simple": {
        "python": ["attention/simple.py"], "go": ["attention/matrix.go", "attention/simple.go"],
        "rust": ["src/attention/matrix.rs", "src/attention/simple.rs"],
        "c": ["attention/matrix.h", "attention/matrix.c", "attention/simple.h", "attention/simple.c"],
        "zig": ["src/attention/matrix.zig", "src/attention/simple.zig"]},
    "attn_self": {
        "python": ["attention/self_attention.py"], "go": ["attention/self_attention.go"],
        "rust": ["src/attention/self_attention.rs"],
        "c": ["attention/self_attention.h", "attention/self_attention.c"], "zig": ["src/attention/self_attention.zig"]},
    "attn_causal": {
        "python": ["attention/causal.py", "attention/rng.py"], "go": ["attention/causal.go", "attention/rng.go"],
        "rust": ["src/attention/causal.rs", "src/tokenization/rng.rs"],
        "c": ["attention/causal.h", "attention/causal.c", "tokenization/rng.h", "tokenization/rng.c"],
        "zig": ["src/attention/causal.zig", "src/tokenization/rng.zig"]},
    "attn_multihead": {
        "python": ["attention/multihead.py"], "go": ["attention/multihead.go"],
        "rust": ["src/attention/multihead.rs"],
        "c": ["attention/multihead.h", "attention/multihead.c"], "zig": ["src/attention/multihead.zig"]},
    "ch03_demo": {
        "python": ["ch03.py"], "go": ["cmd/ch03.go"], "rust": ["src/ch03.rs"],
        "c": ["ch03.c"], "zig": ["src/ch03.zig"]},
    "preprocess": {
        "python": ["tokenization/preprocess.py"], "go": ["tokenization/preprocess.go"],
        "rust": ["src/tokenization/preprocess.rs"],
        "c": ["tokenization/preprocess.h", "tokenization/preprocess.c"], "zig": ["src/tokenization/preprocess.zig"]},
    "vocab_simple": {
        "python": ["tokenization/vocab.py", "tokenization/simple.py"],
        "go": ["tokenization/vocab.go", "tokenization/simple.go"],
        "rust": ["src/tokenization/vocab.rs", "src/tokenization/simple.rs"],
        "c": ["tokenization/vocab.h", "tokenization/vocab.c", "tokenization/simple.h", "tokenization/simple.c"],
        "zig": ["src/tokenization/vocab.zig", "src/tokenization/simple.zig"]},
    "simple": {
        "python": ["tokenization/simple.py"], "go": ["tokenization/simple.go"],
        "rust": ["src/tokenization/simple.rs"],
        "c": ["tokenization/simple.h", "tokenization/simple.c"], "zig": ["src/tokenization/simple.zig"]},
    "bpe": {
        "python": ["main.py"], "go": ["cmd/main.go"], "rust": ["src/main.rs"],
        "c": ["main.c", "tokenization/bpe.h", "tokenization/bpe.c"],
        "zig": ["src/main.zig", "src/tokenization/bpe.zig"]},
    "sampling": {
        "python": ["tokenization/dataset.py"],
        "go": ["tokenization/dataset.go", "tokenization/dataloader.go"],
        "rust": ["src/tokenization/dataset.rs", "src/tokenization/dataloader.rs"],
        "c": ["tokenization/dataset.h", "tokenization/dataset.c", "tokenization/dataloader.h", "tokenization/dataloader.c"],
        "zig": ["src/tokenization/dataset.zig", "src/tokenization/dataloader.zig"]},
    "embedding": {
        "python": ["tokenization/embedding.py"], "go": ["tokenization/embedding.go"],
        "rust": ["src/tokenization/embedding.rs", "src/tokenization/rng.rs"],
        "c": ["tokenization/embedding.h", "tokenization/embedding.c", "tokenization/rng.h", "tokenization/rng.c"],
        "zig": ["src/tokenization/embedding.zig", "src/tokenization/rng.zig"]},
}

BOOK = "Build a Large Language Model (From Scratch), Sebastian Raschka"
BOOK_REPO = "https://github.com/rasbt/LLMs-from-scratch/tree/main"
THIS_REPO = "https://github.com/SobhanYasami/LLMs-from-scratch-by-me"
