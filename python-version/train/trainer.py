"""Step 18 (book 5.2) - the training loop.

One training step is five moves, repeated for every batch:

    1. zero the old gradients            optimizer.zero_grad()
    2. forward pass, measure the loss    loss = calc_loss_batch(...)
    3. backward pass: how should every weight change to lower the loss?   loss.backward()
    4. nudge every weight a little       optimizer.step()      (AdamW)
    5. now and then: report the loss on training and validation data

torch does step 3 for us (autograd). The other four languages in this repo have
no autograd, so they write the backward pass by hand: see gpt/grad.* there.
"""

import tiktoken
import torch

from gpt import generate_text_simple

from .loss import calc_loss_batch, calc_loss_loader


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def evaluate_model(model, train_loader, val_loader, eval_iter: int):
    model.eval()  # dropout off
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context: str, max_new_tokens: int = 20) -> None:
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    out = generate_text_simple(model, encoded, max_new_tokens, context_size)
    print(token_ids_to_text(out, tokenizer).replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs: int, eval_freq: int, eval_iter: int, start_context: str, tokenizer, log=print):
    train_losses, val_losses, tokens_seen_log = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 1
            loss = calc_loss_batch(input_batch, target_batch, model)  # 2
            loss.backward()  # 3
            optimizer.step()  # 4
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:  # 5
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen_log.append(tokens_seen)
                log(f"Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, start_context)  # a sample after every epoch
    return train_losses, val_losses, tokens_seen_log
