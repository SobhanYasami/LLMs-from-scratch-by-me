/* Step 13 (book 4.4) - shortcut connections, and why they matter.
 *
 * A shortcut (residual) connection adds a layer's input to its output:
 *
 *   x = x + layer(x)        instead of        x = layer(x)
 *
 * During training the gradient of the loss must travel backwards through every
 * layer. Without shortcuts it shrinks a little at each layer (the vanishing
 * gradient problem), so the first layers barely learn. With shortcuts the
 * gradient also has a short path straight back, so it stays healthy.
 *
 * Python gets the gradients from torch's autograd. Here there is no autograd,
 * so shortcut.c does the backward pass by hand, in double: the chain rule, layer
 * by layer, from the loss back to the first layer. It reproduces the book's numbers.
 */
#ifndef SHORTCUT_H
#define SHORTCUT_H

#include <stddef.h>

#define EXAMPLE_MAX_WIDTH 3

/* Linear + GELU. w is [out][in] like nn.Linear, flattened row by row. */
typedef struct {
    size_t out, in;
    double w[EXAMPLE_MAX_WIDTH * EXAMPLE_MAX_WIDTH];
    double b[EXAMPLE_MAX_WIDTH];
} ExampleLayer;

typedef struct {
    const ExampleLayer *layers;
    size_t n_layers;
    int use_shortcut;
} ExampleDeepNeuralNetwork;

/* One forward and backward pass (loss = (output - target)^2). Fills `means` with the mean
 * absolute gradient of each layer's weight matrix, first layer first. */
void example_gradient_means(const ExampleDeepNeuralNetwork *net, const double *x, double target, double *means);

#endif
