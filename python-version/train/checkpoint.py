"""Step 20 (book 5.4) - saving and loading a model (and the optimizer).

Training is expensive, so save the weights and reload them later. The book uses
torch.save(model.state_dict()). To keep the file readable by every language in this
repo, we write one plain format instead:

    "LLMCKPT1" | 7 x uint32: vocab, context, emb, heads, layers, qkv_bias, has_optimizer
    then raw little-endian float32 numbers, tensor after tensor, in the order
    model_tensors() lists them (the weights), then the AdamW state if present:
    step (as float32), then exp_avg and exp_avg_sq of every tensor.

A checkpoint written by Python can be loaded by Go, Rust, C or Zig, and the other way round.
"""

import struct

import numpy as np
import torch

from gpt import GPTModel

MAGIC = b"LLMCKPT1"


def model_params(model: GPTModel) -> list[torch.nn.Parameter]:
    """The same list, but the live nn.Parameters (for optimizer state and loading)."""
    out = [model.tok_emb.weight, model.pos_emb.weight]
    for b in model.trf_blocks:
        a = b.att
        out += [a.W_query, a.W_key, a.W_value]
        if a.qkv_bias is not None:
            out += list(a.qkv_bias)
        out += [a.out_weight, a.out_bias, b.norm1.scale, b.norm1.shift]
        out += [b.ff.layers[0].weight, b.ff.layers[0].bias, b.ff.layers[2].weight, b.ff.layers[2].bias, b.norm2.scale, b.norm2.shift]
    out += [model.final_norm.scale, model.final_norm.shift, model.out_head.weight]
    return out




def _is_linear(model: GPTModel, param: torch.Tensor) -> bool:
    linears = [model.out_head.weight]
    for b in model.trf_blocks:
        linears += [b.ff.layers[0].weight, b.ff.layers[2].weight]
    return any(param is w for w in linears)


def save_checkpoint(path: str, model: GPTModel, cfg: dict, optimizer=None) -> None:
    params = model_params(model)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<7I", cfg["vocab_size"], cfg["context_length"], cfg["emb_dim"], cfg["n_heads"], cfg["n_layers"], int(cfg["qkv_bias"]), int(optimizer is not None)))
        for p in params:
            arr = p.detach().T if _is_linear(model, p) else p.detach()
            f.write(arr.contiguous().numpy().astype("<f4").tobytes())
        if optimizer is not None:
            state = [optimizer.state[p] for p in params]
            step = float(state[0]["step"])
            f.write(struct.pack("<f", step))
            for key in ("exp_avg", "exp_avg_sq"):
                for p, s in zip(params, state):
                    arr = s[key].T if _is_linear(model, p) else s[key]
                    f.write(arr.contiguous().numpy().astype("<f4").tobytes())


def load_checkpoint(path: str, model: GPTModel, optimizer=None) -> dict:
    """Fill ``model`` (built from the same config) and, if given, ``optimizer`` from the file."""
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC, "not a checkpoint of this repo"
        vocab, ctx, emb, heads, layers, qkv, has_opt = struct.unpack("<7I", f.read(28))
        params = model_params(model)

        def read_like(p):
            shape = tuple(reversed(p.shape)) if _is_linear(model, p) else tuple(p.shape)
            n = int(np.prod(shape))
            arr = torch.from_numpy(np.frombuffer(f.read(4 * n), dtype="<f4").copy()).reshape(shape)
            return arr.T if _is_linear(model, p) else arr

        with torch.no_grad():
            for p in params:
                p.copy_(read_like(p))
        if optimizer is not None and has_opt:
            step = struct.unpack("<f", f.read(4))[0]
            for key in ("exp_avg", "exp_avg_sq"):
                for p in params:
                    state = optimizer.state[p]
                    if "step" not in state:
                        state["step"] = torch.tensor(step)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state[key].copy_(read_like(p))
    return {"vocab_size": vocab, "context_length": ctx, "emb_dim": emb, "n_heads": heads, "n_layers": layers, "qkv_bias": bool(qkv)}
