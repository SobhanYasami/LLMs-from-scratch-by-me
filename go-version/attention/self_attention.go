package attention

import "math"

// Step 7 (book 3.4) - self-attention with trainable weights (scaled dot-product).
//
// Three weight matrices turn each input vector into three roles:
//
//	query = x x W_query    "what am I looking for?"
//	key   = x x W_key      "what do I offer to a search?"
//	value = x x W_value    "what do I hand over if I am chosen?"
//
//	scores  = query x key^T                 how well does each query match each key?
//	weights = softmax(scores / sqrt(d_k))   scale, then normalise each row
//	context = weights x value               mix the values
//
// Why divide by sqrt(d_k)? Big dot products push softmax towards a hard 0/1
// choice, which makes gradients tiny. Scaling keeps training healthy.
//
// Tip: do not confuse weight parameters (W_*, learned) with attention weights
// (softmax output, computed fresh for every input).

type SelfAttention struct {
	WQuery, WKey, WValue Matrix
}

// NewRandomSelfAttention draws weights uniform in [0, 1), in query, key, value order.
func NewRandomSelfAttention(dIn, dOut int, rng *Rng) *SelfAttention {
	return &SelfAttention{rng.Rand(dIn, dOut), rng.Rand(dIn, dOut), rng.Rand(dIn, dOut)}
}

func scaleBySqrt(scores Matrix, dK int) Matrix {
	return Scale(scores, 1/float32(math.Sqrt(float64(dK))))
}

// AttentionWeights takes x [T, d_in] and returns [T, T] weights.
func (a *SelfAttention) AttentionWeights(x Matrix) Matrix {
	queries, keys := MatMul(x, a.WQuery), MatMul(x, a.WKey)
	scores := MatMul(queries, Transpose(keys))
	return SoftmaxRows(scaleBySqrt(scores, keys.Cols))
}

// Forward returns context vectors [T, d_out].
func (a *SelfAttention) Forward(x Matrix) Matrix {
	return MatMul(a.AttentionWeights(x), MatMul(x, a.WValue))
}
