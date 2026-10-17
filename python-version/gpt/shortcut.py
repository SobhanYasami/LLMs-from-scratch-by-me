"""Step 13 (book 4.4) - shortcut connections, and why they matter.

A shortcut (residual) connection adds a layer's input to its output:

    x = x + layer(x)        instead of        x = layer(x)

During training the gradient of the loss must travel backwards through every
layer. Without shortcuts it shrinks a little at each layer (the vanishing
gradient problem), so the first layers barely learn. With shortcuts the
gradient also has a short path straight back, so it stays healthy.

This tiny 5-layer network lets you see it: print the mean absolute gradient of
each layer's weights with and without shortcuts.
"""

import torch
from torch import nn

from .layers import GELU


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int], use_shortcut: bool):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU()) for i in range(len(layer_sizes) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer_output = layer(x)
            # A shortcut needs matching shapes, so the last (3 -> 1) layer has none.
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

    def gradient_means(self, x: torch.Tensor, target: torch.Tensor) -> list[float]:
        """Mean absolute gradient of each layer's weight matrix, first layer first."""
        self.zero_grad()
        loss = nn.functional.mse_loss(self(x), target)
        loss.backward()
        return [layer[0].weight.grad.abs().mean().item() for layer in self.layers]
