"""Chapter 4 demos: the GPT model, one building block at a time (book 4.1 to 4.7).

Run:  python-version/.venv/bin/python python-version/main.py ch04
      python-version/.venv/bin/python python-version/main.py ch04-124m   (the real 124M model)

Tip: read the demos top to bottom. Numbers in 4.2, 4.3 and 4.4 match the book
(the 4.4 weights are the book's torch.manual_seed(123) ones, written out so
every language shares them). The model demos use random weights from the repo's
shared generator, so all five languages print the same numbers; they differ from
the book's torch output, and since the model is untrained its text is gibberish.
"""

import math

import tiktoken
import torch

from gpt import (
    GPT2_SIZES,
    GPT_CONFIG_124M,
    GPT_CONFIG_TINY,
    ExampleDeepNeuralNetwork,
    FeedForward,
    GELU,
    GPTModel,
    LayerNorm,
    TransformerBlock,
    count_parameters,
    generate_text_simple,
    init_weights,
    parameter_count,
)

# Book 4.1: two short texts, already turned into token IDs by the GPT-2 tokenizer.
BATCH = torch.tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])  # "Every effort moves you" / "Every day holds a"

# Book 4.2: the outputs of a Linear(5, 6) + ReLU layer for two inputs.
LAYER_OUT = torch.tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000], [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]])

# Book 4.4: weights and biases of five Linear layers (3->3 four times, then 3->1), torch.manual_seed(123).
SHORTCUT_LAYERS = [
    ([[-0.23542964, 0.01912448, -0.28674594], [0.21772662, -0.49193421, 0.42322308], [-0.41964141, -0.45901766, -0.36482018]], [0.26147819, -0.21332639, 0.21605217]),
    ([[-0.49001414, -0.35029206, -0.21198919], [-0.11346072, -0.44043937, 0.37804362], [-0.13615717, 0.18532233, 0.40826949]], [0.10756382, 0.15787685, 0.55729234]),
    ([[-0.26039040, 0.18287641, -0.25687242], [0.41260317, 0.46110451, -0.53230095], [0.49285263, 0.27569306, 0.25159022]], [0.23768058, 0.47995073, -0.07623307]),
    ([[-0.48826423, -0.16567004, -0.40661314], [0.03816671, -0.10779562, -0.30968198], [-0.05249349, 0.54698384, -0.04553874]], [0.01833092, -0.09004387, 0.09076351]),
    ([[0.51442641, 0.35304558, 0.20493712]], [0.12546431]),
]


def fmt(v: float) -> str:
    s = f"{v:.4f}"
    return "0.0000" if s == "-0.0000" else s  # a tiny negative number is just zero


def fmt_row(values) -> str:
    return "[" + ", ".join(fmt(v) for v in values) + "]"


def show(label: str, t: torch.Tensor) -> None:
    t = t.detach()
    if t.dim() == 1:
        print(f"{label}: {fmt_row(t.tolist())}")
    else:
        print(f"{label}:")
        for row in t.tolist():
            print("  " + fmt_row(row))


def build(cfg: dict) -> GPTModel:
    model = GPTModel(cfg)
    init_weights(model, seed=123)
    model.eval()  # switch dropout off
    return model


def demo_big_picture() -> None:
    """4.1: token IDs in, one score per vocabulary word out, for every position."""
    cfg = GPT_CONFIG_TINY
    print("== 4.1 the big picture (tiny GPT: emb 32, 4 heads, 2 blocks) ==")
    print(f"batch of token IDs, shape {tuple(BATCH.shape)}: {BATCH.tolist()}")
    logits = build(cfg)(BATCH)
    print(f"logits shape: {tuple(logits.shape)}  <- [batch, tokens, vocab_size]")
    show("logits[0][0], first 4 of 50257", logits[0, 0, :4])


def demo_layer_norm() -> None:
    """4.2: make each token's numbers mean 0, variance 1."""
    print("== 4.2 layer normalization ==")
    mean = LAYER_OUT.mean(dim=-1, keepdim=True)
    var = LAYER_OUT.var(dim=-1, keepdim=True)  # the book first uses the unbiased variance
    show("mean per row", mean.squeeze(-1))
    show("variance per row", var.squeeze(-1))
    show("normalized, row 0", ((LAYER_OUT - mean) / torch.sqrt(var))[0])
    out_ln = LayerNorm(6)(LAYER_OUT)
    show("LayerNorm output mean", out_ln.mean(dim=-1))
    show("LayerNorm output variance (biased)", out_ln.var(dim=-1, unbiased=False))


def demo_gelu_and_feed_forward() -> None:
    """4.3: a smooth ReLU, and the expand-then-shrink feed forward block."""
    print("== 4.3 GELU and the feed forward network ==")
    x = torch.linspace(-3, 3, 7)
    show("x   ", x)
    show("GELU", GELU()(x))
    show("ReLU", torch.relu(x))
    ffn = FeedForward(GPT_CONFIG_TINY["emb_dim"])
    out = ffn(torch.rand(2, 3, GPT_CONFIG_TINY["emb_dim"]))
    print(f"feed forward: input (2, 3, 32) -> output {tuple(out.shape)} (hidden layer is 4 x 32 = 128 wide)")


def demo_shortcut() -> None:
    """4.4: shortcut connections keep the gradient alive."""
    print("== 4.4 shortcut connections (mean |gradient| of each layer's weights) ==")
    x, target = torch.tensor([[1.0, 0.0, -1.0]]), torch.tensor([[0.0]])
    for use_shortcut in (False, True):
        net = ExampleDeepNeuralNetwork([3, 3, 3, 3, 3, 1], use_shortcut)
        with torch.no_grad():
            for layer, (w, b) in zip(net.layers, SHORTCUT_LAYERS):
                layer[0].weight.copy_(torch.tensor(w))
                layer[0].bias.copy_(torch.tensor(b))
        grads = net.gradient_means(x, target)
        print(f"{'with   ' if use_shortcut else 'without'} shortcut: " + "  ".join(f"{g:.6f}" for g in grads) + "   (layer 0 ... layer 4)")


def demo_transformer_block() -> None:
    """4.5: attention + feed forward, each wrapped in norm and shortcut. Shape in = shape out."""
    print("== 4.5 the transformer block ==")
    cfg = GPT_CONFIG_TINY
    model = build(cfg)
    block: TransformerBlock = model.trf_blocks[0]
    x = model.tok_emb(BATCH) + model.pos_emb(torch.arange(4))
    out = block(x)
    print(f"input shape: {tuple(x.shape)}  output shape: {tuple(out.shape)}")
    show("input[0][0], first 4 numbers", x[0, 0, :4])
    show("output[0][0], first 4 numbers", out[0, 0, :4])


def demo_parameters() -> None:
    """4.6: how big is the model?"""
    print("== 4.6 model size ==")
    tiny = build(GPT_CONFIG_TINY)
    print(f"tiny model: {count_parameters(tiny):,} parameters (arithmetic says {parameter_count(GPT_CONFIG_TINY):,})")
    total = parameter_count(GPT_CONFIG_124M)
    head = GPT_CONFIG_124M["vocab_size"] * GPT_CONFIG_124M["emb_dim"]
    print(f"GPT-2 small, no model built: {total:,} parameters")
    print(f"  minus the output head (weight tying in the original GPT-2): {total - head:,}")
    print(f"  float32 size: {total * 4 / (1024 * 1024):.2f} MB")
    d = GPT_CONFIG_124M["emb_dim"]
    ff = d * 4 * d + 4 * d + 4 * d * d + d
    att = 4 * d * d + d
    print(f"exercise 4.1: feed forward {ff:,} vs attention {att:,} parameters per block")
    print("exercise 4.2: bigger models, same code, different configuration")
    print(f"  {'':<20} {'parameters':>14} {'weight-tied':>14} {'float32 MB':>11}")
    for name, sizes in GPT2_SIZES.items():
        cfg = {**GPT_CONFIG_124M, **sizes}
        n = parameter_count(cfg)
        tied = n - cfg["vocab_size"] * cfg["emb_dim"]
        print(f"  {name:<20} {n:>14,} {tied:>14,} {n * 4 / (1024 * 1024):>11,.2f}")


def demo_generate(cfg: dict, label: str, max_new_tokens: int = 6) -> None:
    """4.7: predict one token, append it, repeat."""
    print(f"== 4.7 generating text ({label}) ==")
    model = build(cfg)
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode("Hello, I am")
    print(f"encoded: {encoded}")
    out = generate_text_simple(model, torch.tensor(encoded).unsqueeze(0), max_new_tokens, cfg["context_length"])
    print(f"output IDs: {out[0].tolist()}")
    print(f"output length: {out.shape[1]}")
    print(f"decoded: {tokenizer.decode(out[0].tolist())}")
    print("(gibberish: the weights are random, nothing has been trained yet)")


def run() -> None:
    demo_big_picture()
    demo_layer_norm()
    demo_gelu_and_feed_forward()
    demo_shortcut()
    demo_transformer_block()
    demo_parameters()
    demo_generate(GPT_CONFIG_TINY, "tiny model")


def run_124m() -> None:
    """The real 124M-parameter model: forward pass, then generation. Needs ~1 GB of memory."""
    model = build(GPT_CONFIG_124M)
    print("== GPT-2 small (124M) ==")
    print(f"parameters built: {count_parameters(model):,}")
    with torch.no_grad():
        logits = model(BATCH)
    print(f"logits shape: {tuple(logits.shape)}")
    show("logits[0][0], first 4 of 50257", logits[0, 0, :4])
    demo_generate(GPT_CONFIG_124M, "124M model")
