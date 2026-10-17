"""Pretraining (book chapter 5).

Files, in the order you learn them:

- ``loss``: cross entropy loss and perplexity (5.1).
- ``trainer``: the training loop, evaluation and sample printing (5.2).
- ``sample``: temperature and top-k decoding (5.3).
- ``checkpoint``: saving and loading weights and optimizer state (5.4).
- ``openai``: loading OpenAI's pretrained GPT-2 weights (5.5).
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .loss import calc_loss_batch, calc_loss_loader, cross_entropy, perplexity
from .openai import GPT2_SMALL_CONFIG, load_gpt2_weights, read_safetensors
from .sample import generate, sample_index, softmax_with_temperature, top_k_logits
from .trainer import evaluate_model, generate_and_print_sample, text_to_token_ids, token_ids_to_text, train_model_simple

__all__ = [
    "GPT2_SMALL_CONFIG", "calc_loss_batch", "calc_loss_loader", "cross_entropy", "evaluate_model", "generate",
    "generate_and_print_sample", "load_checkpoint", "load_gpt2_weights", "perplexity", "read_safetensors",
    "sample_index", "save_checkpoint", "softmax_with_temperature", "text_to_token_ids", "token_ids_to_text",
    "top_k_logits", "train_model_simple",
]
