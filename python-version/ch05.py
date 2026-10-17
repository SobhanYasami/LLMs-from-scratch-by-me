"""Chapter 5 demos: pretraining, one step at a time (book 5.1 to 5.5).

Run:  python-version/.venv/bin/python python-version/main.py ch05
      python-version/.venv/bin/python python-version/main.py ch05-gpt2   (OpenAI's weights, 5.5)

Tip: read the demos top to bottom. The model is the tiny GPT of chapter 4 (emb 32, 2 blocks,
context 16) so that every language can TRAIN it in seconds; the book trains the 124M model
on a GPU. All five languages use the same weights, data order (no shuffling) and sampling
generator, so they print the same numbers. They differ from the book's torch output.
"""

import math
from pathlib import Path

import tiktoken
import torch

from attention import Rng
from gpt import GPT_CONFIG_TINY, GPTModel, generate_text_simple, init_weights
from tokenization.dataset import create_dataloader_v1
from train import (
    GPT2_SMALL_CONFIG,
    calc_loss_batch,
    calc_loss_loader,
    cross_entropy,
    generate,
    load_checkpoint,
    load_gpt2_weights,
    perplexity,
    read_safetensors,
    save_checkpoint,
    sample_index,
    softmax_with_temperature,
    text_to_token_ids,
    token_ids_to_text,
    top_k_logits,
    train_model_simple,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CFG = {**GPT_CONFIG_TINY, "drop_rate": 0.0}  # dropout stays off: our loop has no random masks
BATCH, STRIDE, EPOCHS, EVAL_FREQ, EVAL_ITER, LR, WEIGHT_DECAY = 4, 16, 2, 24, 2, 3e-3, 0.1


def fmt(v: float, digits: int = 4) -> str:
    s = f"{v:.{digits}f}"
    return f"{0:.{digits}f}" if s == f"-{0:.{digits}f}" else s


def quoted(text: str) -> str:
    """Text in single quotes with newlines flattened (same in every language)."""
    return "'" + text.replace("\n", " ") + "'"


def fmt_row(values, digits: int = 4) -> str:
    return "[" + ", ".join(fmt(v, digits) for v in values) + "]"


def build(cfg=CFG) -> GPTModel:
    model = GPTModel(cfg)
    init_weights(model, seed=123)
    return model


def loaders(tokenizer):
    text = (ASSETS / "the-verdict.txt").read_text(encoding="utf-8")
    split = int(0.9 * len(text))  # 90% training, 10% validation
    train = create_dataloader_v1(text[:split], tokenizer, batch_size=BATCH, max_length=CFG["context_length"], stride=STRIDE, shuffle=False, drop_last=True)
    val = create_dataloader_v1(text[split:], tokenizer, batch_size=BATCH, max_length=CFG["context_length"], stride=STRIDE, shuffle=False, drop_last=False)
    return train, val


def demo_loss(tokenizer) -> None:
    """5.1: from probabilities to a single number."""
    print("== 5.1 evaluating generative text models ==")
    model = build().eval()
    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])  # ["every effort moves", "I really like"]
    targets = torch.tensor([[3626, 6100, 345], [588, 428, 11311]])  # [" effort moves you", " really like chocolate"]
    print(f"inputs : {inputs.tolist()}")
    print(f"targets: {targets.tolist()}")
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print(f"probas shape: {tuple(probas.shape)}  <- [batch, tokens, vocab_size]")
    picked = [probas[i, [0, 1, 2], targets[i]] for i in range(2)]  # the probability given to each correct next token
    for i, p in enumerate(picked):
        print(f"probability of the target, text {i + 1}: {fmt_row(p.tolist(), 6)}")
    log_probas = torch.log(torch.cat(picked))
    print(f"log probabilities: {fmt_row(log_probas.tolist())}")
    avg = log_probas.mean().item()
    print(f"average log probability: {fmt(avg)}")
    print(f"negative average (the loss): {fmt(-avg)}")
    loss = cross_entropy(logits, targets).item()
    print(f"cross_entropy of the whole batch at once: {fmt(loss)}")
    print(f"perplexity: {fmt(perplexity(loss), 0)}  (a model that guesses uniformly is torn between 50,257 tokens)")
    train_loader, val_loader = loaders(tokenizer)
    print(f"training batches: {len(train_loader)} of {BATCH} x {CFG['context_length']} tokens, validation batches: {len(val_loader)}")
    with torch.no_grad():
        tl, vl = calc_loss_loader(train_loader, model), calc_loss_loader(val_loader, model)
    print(f"untrained: training loss {fmt(tl, 3)}, validation loss {fmt(vl, 3)}  (a uniform guess gives ln(50257) = {fmt(math.log(50257), 3)})")


def train_demo(tokenizer):
    """5.2: train for a few epochs. Returns the trained model and the optimizer."""
    print("== 5.2 training an LLM ==")
    model = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader, val_loader = loaders(tokenizer)
    train_model_simple(model, train_loader, val_loader, optimizer, EPOCHS, EVAL_FREQ, EVAL_ITER, "Every effort moves you", tokenizer)
    return model, optimizer


VOCAB = ["closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you"]
NEXT_TOKEN_LOGITS = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])


def demo_sampling(model, tokenizer) -> None:
    """5.3: temperature and top-k."""
    print("== 5.3 decoding strategies ==")
    probas = torch.softmax(NEXT_TOKEN_LOGITS, dim=0)
    print(f"next-token probabilities: {fmt_row(probas.tolist())}")
    print(f"greedy pick: {VOCAB[int(torch.argmax(probas))]}")
    for t in (1.0, 0.1, 5.0):
        p = softmax_with_temperature(NEXT_TOKEN_LOGITS, t)
        rng = Rng(123)
        counts = [0] * len(VOCAB)
        for _ in range(1000):
            counts[sample_index(p, rng)] += 1
        shown = ", ".join(f"{c} x {w}" for c, w in zip(counts, VOCAB) if c)
        print(f"temperature {t:g}: P(pizza) = {fmt(float(p[6]))}  1000 draws: {shown}")
    top = torch.topk(NEXT_TOKEN_LOGITS, 3)
    print(f"top-3 logits: {fmt_row(top.values.tolist(), 2)} at positions {top.indices.tolist()}")
    masked = top_k_logits(NEXT_TOKEN_LOGITS, 3)
    print(f"after masking with -inf: {[('-inf' if math.isinf(v) else fmt(v, 2)) for v in masked.tolist()]}")
    print(f"top-3 probabilities: {fmt_row(torch.softmax(masked, dim=0).tolist())}")
    model.eval()
    start = text_to_token_ids("Every effort moves you", tokenizer)
    context = CFG["context_length"]
    greedy = generate_text_simple(model, start, 15, context)
    print(f"greedy   : {quoted(token_ids_to_text(greedy, tokenizer))}")
    sampled = generate(model, start, 15, context, Rng(123), temperature=1.4, top_k=25)
    print(f"top-k 25, temperature 1.4: {quoted(token_ids_to_text(sampled, tokenizer))}")
    same = generate(model, start, 15, context, Rng(123), temperature=0.0, top_k=None)
    print(f"temperature 0 equals greedy: {str(bool(torch.equal(same, greedy))).lower()}")


def demo_checkpoint(model, optimizer, tokenizer) -> None:
    """5.4: save, load, and carry on exactly where we stopped."""
    print("== 5.4 saving and loading ==")
    path = ASSETS / "checkpoints" / "python.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(path), model, CFG, optimizer)
    print(f"saved model and optimizer state: {path.stat().st_size:,} bytes")
    restored = build({**CFG})
    for p in restored.parameters():  # start from garbage so a successful load is visible
        torch.nn.init.constant_(p, 0.5)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    load_checkpoint(str(path), restored, restored_opt)
    x = torch.tensor([[6109, 3626, 6100, 345]])
    with torch.no_grad():
        diff = (model(x) - restored(x)).abs().max().item()
    print(f"largest logit difference between the two models: {fmt(diff, 6)}")
    train_loader, _ = loaders(tokenizer)
    losses = {}
    for name, m, opt in (("continued", model, optimizer), ("restored ", restored, restored_opt)):
        run = []
        for i, (xb, yb) in enumerate(train_loader):
            if i == 3:
                break
            opt.zero_grad()
            loss = calc_loss_batch(xb, yb, m)
            loss.backward()
            opt.step()
            run.append(loss.item())
        losses[name] = run
        print(f"3 more steps, {name}: losses {fmt_row(run, 3)}")
    print(f"same losses: {str(losses['continued'] == losses['restored ']).lower()}")


def run() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    demo_loss(tokenizer)
    model, optimizer = train_demo(tokenizer)
    demo_sampling(model, tokenizer)
    demo_checkpoint(model, optimizer, tokenizer)


def run_gpt2() -> None:
    """5.5: OpenAI's GPT-2 small, loaded into our model. Needs assets/gpt2/model.safetensors (see train/openai.py)."""
    tokenizer = tiktoken.get_encoding("gpt2")
    path = ASSETS / "gpt2" / "model.safetensors"
    if not path.exists():
        raise SystemExit(f"missing {path}: download it first (command in python-version/train/openai.py)")
    print("== 5.5 loading pretrained weights from OpenAI ==")
    model = GPTModel(GPT2_SMALL_CONFIG)
    load_gpt2_weights(model, read_safetensors(str(path)))
    model.eval()
    print(f"loaded GPT-2 small: {sum(p.numel() for p in model.parameters()):,} parameters (with the attention biases)")
    start = text_to_token_ids("Every effort moves you", tokenizer)
    out = generate(model, start, 25, GPT2_SMALL_CONFIG["context_length"], Rng(123), temperature=1.5, top_k=50)
    print(f"generated: {quoted(token_ids_to_text(out, tokenizer))}")
    greedy = generate_text_simple(model, start, 25, GPT2_SMALL_CONFIG["context_length"])
    print(f"greedy   : {quoted(token_ids_to_text(greedy, tokenizer))}")
    text = (ASSETS / "the-verdict.txt").read_text(encoding="utf-8")
    split = int(0.9 * len(text))
    val = create_dataloader_v1(text[split:], tokenizer, batch_size=2, max_length=64, stride=64, shuffle=False, drop_last=True)
    with torch.no_grad():
        loss = calc_loss_loader(val, model, num_batches=2)
    print(f"exercise 5.5: loss of the pretrained model on The Verdict (validation part, 2 batches of 2 x 64 tokens): {fmt(loss, 3)}")
