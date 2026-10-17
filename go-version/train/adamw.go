package train

// The AdamW optimizer (book 5.2), the same arithmetic as torch.optim.AdamW.
//
// Plain gradient descent moves every weight by lr * gradient. Adam keeps two running averages per
// weight, so each weight gets its own step size:
//
//	m = b1*m + (1-b1)*g          average of the gradient         (the direction)
//	v = b2*v + (1-b2)*g*g        average of the squared gradient (how noisy it is)
//	w = w - lr * m_hat / (sqrt(v_hat) + eps)
//
// "W" means the weight decay is applied straight to the weights (w *= 1 - lr*wd) instead of being
// mixed into the gradient, which regularizes better.

import (
	"math"

	"llm-from-scratch/gpt"
)

type AdamW struct {
	// The settings are float64 like Python's floats; the per-step scalars are rounded to float32 once.
	LR, Beta1, Beta2, Eps, WeightDecay float64
	Step                               int
	// Running averages, stored in models of the same shape as the one being trained.
	ExpAvg, ExpAvgSq *gpt.Model
}

func NewAdamW(m *gpt.Model, lr, weightDecay float64) *AdamW {
	return &AdamW{LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: weightDecay, ExpAvg: m.ZerosLike(), ExpAvgSq: m.ZerosLike()}
}

// StepWith updates every weight in m from grads (a model-shaped bag of gradients).
func (o *AdamW) StepWith(m, grads *gpt.Model) {
	o.Step++
	t := float64(o.Step)
	stepSize := float32(o.LR / (1 - math.Pow(o.Beta1, t)))
	bias2Sqrt := float32(math.Sqrt(1 - math.Pow(o.Beta2, t)))
	decay := float32(1 - o.LR*o.WeightDecay)
	b2, eps := float32(o.Beta2), float32(o.Eps)
	oneMinusB1, oneMinusB2 := float32(1-o.Beta1), float32(1-o.Beta2)
	params, ms, vs, gs := m.Tensors(), o.ExpAvg.Tensors(), o.ExpAvgSq.Tensors(), grads.Tensors()
	for k, p := range params {
		mk, vk, gk := ms[k], vs[k], gs[k]
		for i := range p {
			p[i] *= decay
			mk[i] += (gk[i] - mk[i]) * oneMinusB1 // lerp
			vk[i] = vk[i]*b2 + oneMinusB2*gk[i]*gk[i]
			denom := float32(math.Sqrt(float64(vk[i])))/bias2Sqrt + eps
			p[i] += (-stepSize * mk[i]) / denom
		}
	}
}
