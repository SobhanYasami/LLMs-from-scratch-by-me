//! The AdamW optimizer (book 5.2), the same arithmetic as `torch.optim.AdamW`.
//!
//! Plain gradient descent moves every weight by `lr * gradient`. Adam keeps two running
//! averages per weight, so each weight gets its own step size:
//!
//! ```text
//! m = b1*m + (1-b1)*g          average of the gradient        (the direction)
//! v = b2*v + (1-b2)*g*g        average of the squared gradient (how noisy it is)
//! w = w - lr * m_hat / (sqrt(v_hat) + eps)
//! ```
//!
//! "W" means the weight decay is applied straight to the weights (`w *= 1 - lr*wd`) instead of
//! being mixed into the gradient, which regularizes better.

use crate::gpt::GptModel;

pub struct AdamW {
    // The settings are f64 like Python's floats; the per-step scalars are rounded to f32 once.
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub step: u32,
    /// Running averages, stored in models of the same shape as the one being trained.
    pub exp_avg: GptModel,
    pub exp_avg_sq: GptModel,
}

impl AdamW {
    pub fn new(model: &GptModel, lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step: 0,
            exp_avg: model.zeros_like(),
            exp_avg_sq: model.zeros_like(),
        }
    }

    /// One update of every weight in `model` from `grads` (a model-shaped bag of gradients).
    pub fn step(&mut self, model: &mut GptModel, grads: &GptModel) {
        self.step += 1;
        let t = self.step as i32;
        let step_size = (self.lr / (1.0 - self.beta1.powi(t))) as f32;
        let bias2_sqrt = (1.0 - self.beta2.powi(t)).sqrt() as f32;
        let decay = (1.0 - self.lr * self.weight_decay) as f32;
        let (b2, eps) = (self.beta2 as f32, self.eps as f32);
        let (one_minus_b1, one_minus_b2) = ((1.0 - self.beta1) as f32, (1.0 - self.beta2) as f32);
        let params = model.tensors_mut();
        let ms = self.exp_avg.tensors_mut();
        let vs = self.exp_avg_sq.tensors_mut();
        for (((p, m), v), g) in params.into_iter().zip(ms).zip(vs).zip(grads.tensors()) {
            for i in 0..p.len() {
                p[i] *= decay;
                m[i] += (g[i] - m[i]) * one_minus_b1; // lerp
                v[i] = v[i] * b2 + one_minus_b2 * g[i] * g[i];
                let denom = v[i].sqrt() / bias2_sqrt + eps;
                p[i] += (-step_size * m[i]) / denom;
            }
        }
    }
}
