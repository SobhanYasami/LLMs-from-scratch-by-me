"""Step 4 - Sliding-window data sampling for next-token prediction.

Given a long list of token IDs, cut it into overlapping windows:

    ids     = [10, 11, 12, 13, 14, 15]
    window 0: input [10, 11, 12, 13]  target [11, 12, 13, 14]
    window 1: input [12, 13, 14, 15]  target [13, 14, 15, 16]   (if it fits)

Why shifted by one? The model's job is: given the tokens so far, guess the
next one, so every position in a window is a free training example.
The target is the input shifted by one position. ``stride`` is how far the
window moves each step; ``stride == max_length`` means no overlap.
"""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    """Holds every (input, target) window of one tokenized text."""

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        # Stop at len - max_length: the target reaches one token past the input.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    """Wrap ``GPTDatasetV1`` in a batching ``DataLoader``.

    Why batch? The GPU processes many windows at once. Why shuffle? Random order
    smooths training. Why ``drop_last``? A short final batch can upset training.
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
