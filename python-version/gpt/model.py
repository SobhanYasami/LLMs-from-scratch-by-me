"""Steps 15-16 (book 4.6 and 4.7) - the GPT model and text generation.

    token IDs [B, T]
      -> token embedding + position embedding        [B, T, D]
      -> N x transformer block                       [B, T, D]
      -> final layer norm                            [B, T, D]
      -> output head (Linear D -> vocab, no bias)    [B, T, vocab]   = "logits"

The logits of the LAST position are the model's scores for the next token.
``generate_text_simple`` picks the best one, appends it, and repeats.

Weights here start as random numbers from the repo's shared generator
(Normal(0, 0.02), biases 0, layer norm scale 1 / shift 0), in a fixed order,
so every language builds exactly the same untrained model. Untrained means
the generated text is gibberish; training is chapter 5.
"""

import torch
from torch import nn

from attention import Rng

from .block import TransformerBlock
from .layers import LayerNorm

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # BPE vocabulary of GPT-2
    "context_length": 1024,  # most tokens the model can look at
    "emb_dim": 768,  # size of every token vector
    "n_heads": 12,  # attention heads per block
    "n_layers": 12,  # transformer blocks
    "drop_rate": 0.1,  # dropout (training only)
    "qkv_bias": False,  # bias in the query/key/value projections
}

# Same architecture, small enough to run in a blink in any language.
GPT_CONFIG_TINY = {**GPT_CONFIG_124M, "context_length": 16, "emb_dim": 32, "n_heads": 4, "n_layers": 2}

# Book exercise 4.2: bigger GPT-2 models differ only in the configuration.
GPT2_SIZES = {
    "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # [batch, tokens, emb]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def init_weights(model: GPTModel, seed: int = 123, std: float = 0.02) -> None:
    """Fill the model from the shared generator. The ORDER is part of the recipe:
    token emb, position emb, then per block wq, wk, wv, wo, ff1, ff2, then the output head."""
    rng = Rng(seed)

    def normal(shape):  # our matrices are [in, out]; nn.Linear stores [out, in]
        return rng.normal(shape[0], shape[1], std)

    with torch.no_grad():
        model.tok_emb.weight.copy_(normal(model.tok_emb.weight.shape))
        model.pos_emb.weight.copy_(normal(model.pos_emb.weight.shape))
        for block in model.trf_blocks:
            for w in (block.att.W_query, block.att.W_key, block.att.W_value, block.att.out_weight):
                w.copy_(normal(w.shape))
            block.ff.layers[0].weight.copy_(normal(tuple(reversed(block.ff.layers[0].weight.shape))).T)
            block.ff.layers[2].weight.copy_(normal(tuple(reversed(block.ff.layers[2].weight.shape))).T)
            for lin in (block.ff.layers[0], block.ff.layers[2]):
                lin.bias.zero_()
            block.att.out_bias.zero_()
        model.out_head.weight.copy_(normal(tuple(reversed(model.out_head.weight.shape))).T)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_count(cfg: dict) -> int:
    """Parameters of a GPTModel, by arithmetic only (no model is built)."""
    v, c, d, n = cfg["vocab_size"], cfg["context_length"], cfg["emb_dim"], cfg["n_layers"]
    attention = 4 * d * d + d + (3 * d if cfg["qkv_bias"] else 0)  # wq, wk, wv, wo, the output bias (+ qkv biases)
    feed_forward = d * 4 * d + 4 * d + 4 * d * d + d
    norms = 2 * 2 * d  # two layer norms per block, scale + shift each
    return v * d + c * d + n * (attention + feed_forward + norms) + 2 * d + d * v


@torch.no_grad()
def generate_text_simple(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """Greedy decoding. ``idx`` is [batch, tokens] of IDs; returns it with new IDs appended."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # keep only what fits in the context
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # scores for the NEXT token only: [batch, vocab]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # greedy: the most likely token
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
