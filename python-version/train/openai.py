"""Step 21 (book 5.5) - loading OpenAI's pretrained GPT-2 weights.

Training a GPT-2 from scratch costs a fortune; OpenAI published the result. The weights
come as a ``.safetensors`` file, a very plain format: 8 bytes with the header length,
a JSON header that says where each tensor lives, then raw float32 numbers. No library needed.

Download once (about 500 MB, the file is ignored by git):

    curl -L -o assets/gpt2/model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors

OpenAI's layout differs from ours in a few ways, which is what ``load_gpt2_weights`` fixes:

- one big ``c_attn`` matrix holds query, key and value side by side (we keep three);
- the attention projections carry biases (``qkv_bias = True``);
- matrices are stored [in, out]; torch's ``nn.Linear`` wants [out, in], so those get transposed;
- the output head re-uses the token embedding (weight tying), so we copy ``wte`` into it.
"""

import json
import struct

import numpy as np
import torch

from gpt import GPT_CONFIG_124M, GPTModel

GPT2_SMALL_CONFIG = {**GPT_CONFIG_124M, "context_length": 1024, "qkv_bias": True, "drop_rate": 0.0}


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Every tensor of the file as a numpy array (read lazily from disk)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    base = 8 + header_len
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32", f"{name}: only float32 is supported"
        start, end = info["data_offsets"]
        out[name] = np.memmap(path, dtype="<f4", mode="r", offset=base + start, shape=tuple(info["shape"]))
    return out


def assign(param: torch.nn.Parameter, value) -> None:
    value = torch.from_numpy(np.array(value, dtype=np.float32))
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch: model {tuple(param.shape)}, file {tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value)


def load_gpt2_weights(model: GPTModel, t: dict[str, np.ndarray]) -> None:
    assign(model.tok_emb.weight, t["wte.weight"])
    assign(model.pos_emb.weight, t["wpe.weight"])
    for i, block in enumerate(model.trf_blocks):
        p = f"h.{i}."
        att = block.att
        qw, kw, vw = np.split(t[p + "attn.c_attn.weight"], 3, axis=-1)
        qb, kb, vb = np.split(t[p + "attn.c_attn.bias"], 3, axis=-1)
        for param, value in ((att.W_query, qw), (att.W_key, kw), (att.W_value, vw), (att.qkv_bias[0], qb), (att.qkv_bias[1], kb), (att.qkv_bias[2], vb)):
            assign(param, value)
        assign(att.out_weight, t[p + "attn.c_proj.weight"])
        assign(att.out_bias, t[p + "attn.c_proj.bias"])
        assign(block.ff.layers[0].weight, t[p + "mlp.c_fc.weight"].T)
        assign(block.ff.layers[0].bias, t[p + "mlp.c_fc.bias"])
        assign(block.ff.layers[2].weight, t[p + "mlp.c_proj.weight"].T)
        assign(block.ff.layers[2].bias, t[p + "mlp.c_proj.bias"])
        assign(block.norm1.scale, t[p + "ln_1.weight"])
        assign(block.norm1.shift, t[p + "ln_1.bias"])
        assign(block.norm2.scale, t[p + "ln_2.weight"])
        assign(block.norm2.shift, t[p + "ln_2.bias"])
    assign(model.final_norm.scale, t["ln_f.weight"])
    assign(model.final_norm.shift, t["ln_f.bias"])
    assign(model.out_head.weight, t["wte.weight"])  # weight tying: the head is the embedding again
