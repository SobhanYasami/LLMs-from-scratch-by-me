#include "shortcut.h"

#include <math.h>

static double gelu64(double v) { return 0.5 * v * (1.0 + tanh(sqrt(2.0 / M_PI) * (v + 0.044715 * v * v * v))); }

/* The derivative of the tanh-approximated GELU. */
static double gelu_grad(double v) {
    double c = sqrt(2.0 / M_PI);
    double t = tanh(c * (v + 0.044715 * v * v * v));
    return 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * v * v);
}

#define MAX_LAYERS 8

void example_gradient_means(const ExampleDeepNeuralNetwork *net, const double *x, double target, double *means) {
    double acts[MAX_LAYERS + 1][EXAMPLE_MAX_WIDTH]; /* input of every layer */
    double pre[MAX_LAYERS][EXAMPLE_MAX_WIDTH];      /* Linear output (before GELU) of every layer */
    int shortcut[MAX_LAYERS];
    for (size_t j = 0; j < net->layers[0].in; j++) acts[0][j] = x[j];

    for (size_t li = 0; li < net->n_layers; li++) {
        const ExampleLayer *l = &net->layers[li];
        for (size_t i = 0; i < l->out; i++) {
            double z = l->b[i];
            for (size_t j = 0; j < l->in; j++) z += l->w[i * l->in + j] * acts[li][j];
            pre[li][i] = z;
            acts[li + 1][i] = gelu64(z);
        }
        /* A shortcut needs matching shapes, so the last (3 -> 1) layer has none. */
        shortcut[li] = net->use_shortcut && l->out == l->in;
        if (shortcut[li])
            for (size_t i = 0; i < l->out; i++) acts[li + 1][i] += acts[li][i];
    }

    double grad[EXAMPLE_MAX_WIDTH] = {2.0 * (acts[net->n_layers][0] - target)}; /* d loss / d output */
    for (size_t li = net->n_layers; li-- > 0;) {
        const ExampleLayer *l = &net->layers[li];
        double dz[EXAMPLE_MAX_WIDTH]; /* gradient at the Linear output */
        for (size_t i = 0; i < l->out; i++) dz[i] = grad[i] * gelu_grad(pre[li][i]);
        double sum = 0;
        for (size_t i = 0; i < l->out; i++)
            for (size_t j = 0; j < l->in; j++) sum += fabs(dz[i] * acts[li][j]); /* d loss / d W[i][j] */
        means[li] = sum / (double)(l->out * l->in);
        double next[EXAMPLE_MAX_WIDTH] = {0}; /* gradient handed to the layer below */
        for (size_t j = 0; j < l->in; j++) {
            for (size_t i = 0; i < l->out; i++) next[j] += l->w[i * l->in + j] * dz[i];
            if (shortcut[li]) next[j] += grad[j]; /* the shortcut: gradient also flows straight through */
        }
        for (size_t j = 0; j < l->in; j++) grad[j] = next[j];
    }
}
