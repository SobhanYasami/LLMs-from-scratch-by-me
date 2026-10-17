#include "adamw.h"

#include <math.h>

AdamW adamw_new(Arena *a, const GptModel *m, double lr, double weight_decay) {
    return (AdamW){lr, 0.9, 0.999, 1e-8, weight_decay, 0, gpt_model_zeros_like(a, m), gpt_model_zeros_like(a, m)};
}

void adamw_step(AdamW *o, GptModel *m, const GptModel *grads) {
    o->step++;
    double t = (double)o->step;
    float step_size = (float)(o->lr / (1 - pow(o->beta1, t)));
    float bias2_sqrt = (float)sqrt(1 - pow(o->beta2, t));
    float decay = (float)(1 - o->lr * o->weight_decay);
    float b2 = (float)o->beta2, eps = (float)o->eps;
    float one_minus_b1 = (float)(1 - o->beta1), one_minus_b2 = (float)(1 - o->beta2);
    float *p[GPT_MAX_TENSORS], *mm[GPT_MAX_TENSORS], *vv[GPT_MAX_TENSORS], *g[GPT_MAX_TENSORS];
    size_t len[GPT_MAX_TENSORS], n = gpt_tensors(m, p, len);
    gpt_tensors(&o->exp_avg, mm, len);
    gpt_tensors(&o->exp_avg_sq, vv, len);
    gpt_tensors(grads, g, len);
    for (size_t k = 0; k < n; k++)
        for (size_t i = 0; i < len[k]; i++) {
            p[k][i] *= decay;
            mm[k][i] += (g[k][i] - mm[k][i]) * one_minus_b1; /* lerp */
            vv[k][i] = vv[k][i] * b2 + one_minus_b2 * g[k][i] * g[k][i];
            float denom = (float)sqrt((double)vv[k][i]) / bias2_sqrt + eps;
            p[k][i] += (-step_size * mm[k][i]) / denom;
        }
}
