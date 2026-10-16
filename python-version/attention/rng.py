"""Tiny seeded random generator (splitmix64), so every language gives the same numbers.

Used for random weights and for dropout masks. Same seed, same numbers, on
every run and in every language of this repo (Go, Rust, C and Zig have the
same few lines).
"""

import torch

MASK = (1 << 64) - 1


class Rng:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def next_f64(self) -> float:
        """Uniform in [0, 1)."""
        return (self.next_u64() >> 11) / 9007199254740992.0  # 2**53

    def rand(self, rows: int, cols: int) -> torch.Tensor:
        """A [rows, cols] float32 matrix, uniform in [0, 1), like ``torch.rand``."""
        values = [self.next_f64() for _ in range(rows * cols)]
        return torch.tensor(values, dtype=torch.float32).reshape(rows, cols)


def dropout(x: torch.Tensor, p: float, rng: Rng) -> torch.Tensor:
    """Zero each element with probability ``p`` and scale the rest by 1/(1-p).

    The scaling keeps the average value unchanged, so nothing needs rescaling
    at inference time (when dropout is simply switched off).
    """
    keep = [1.0 if rng.next_f64() >= p else 0.0 for _ in range(x.numel())]
    mask = torch.tensor(keep, dtype=x.dtype).reshape(x.shape)
    return x * mask / (1.0 - p)
