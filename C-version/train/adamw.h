/* The AdamW optimizer (book 5.2), the same arithmetic as torch.optim.AdamW.
 *
 * Plain gradient descent moves every weight by lr * gradient. Adam keeps two running averages per
 * weight, so each weight gets its own step size:
 *
 *   m = b1*m + (1-b1)*g          average of the gradient         (the direction)
 *   v = b2*v + (1-b2)*g*g        average of the squared gradient (how noisy it is)
 *   w = w - lr * m_hat / (sqrt(v_hat) + eps)
 *
 * "W" means the weight decay is applied straight to the weights (w *= 1 - lr*wd) instead of being
 * mixed into the gradient, which regularizes better.
 */
#ifndef ADAMW_H
#define ADAMW_H

#include "../gpt/model.h"

typedef struct {
    /* The settings are double like Python's floats; the per-step scalars are rounded to float once. */
    double lr, beta1, beta2, eps, weight_decay;
    int step;
    GptModel exp_avg, exp_avg_sq; /* running averages, in models of the same shape as the one trained */
} AdamW;

AdamW adamw_new(Arena *a, const GptModel *m, double lr, double weight_decay);
/* One update of every weight in `m` from `grads` (a model-shaped bag of gradients). */
void adamw_step(AdamW *o, GptModel *m, const GptModel *grads);

#endif
