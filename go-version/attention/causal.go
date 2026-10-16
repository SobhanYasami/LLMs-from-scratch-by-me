package attention

// Step 8 (book 3.5) - causal attention: hide the future.
//
// A GPT writes text left to right, so token i may only look at tokens 0..i.
// We hide the later tokens with a mask. Set their scores to -Inf before
// softmax: exp(-Inf) = 0, and the remaining weights in each row still sum to 1.
//
//	scores (T x T)                       after the mask
//	[ s00 s01 s02 ]                      [ s00 -inf -inf ]
//	[ s10 s11 s12 ]      -------->       [ s10  s11 -inf ]
//	[ s20 s21 s22 ]                      [ s20  s21  s22 ]
//
// Dropout (training only) randomly zeroes some attention weights so the model
// does not lean on any single connection.

type CausalAttention struct {
	SelfAttention
	DropoutP float32
	Rng      *Rng
}

func NewCausalAttention(wq, wk, wv Matrix, dropoutP float32, rng *Rng) *CausalAttention {
	return &CausalAttention{SelfAttention{wq, wk, wv}, dropoutP, rng}
}

// CausalWeights takes x [T, d_in]. With train set, dropout is applied.
func (a *CausalAttention) CausalWeights(x Matrix, train bool) Matrix {
	queries, keys := MatMul(x, a.WQuery), MatMul(x, a.WKey)
	scores := MaskFuture(MatMul(queries, Transpose(keys)))
	weights := SoftmaxRows(scaleBySqrt(scores, keys.Cols))
	if train && a.DropoutP > 0 {
		weights = Dropout(weights, a.DropoutP, a.Rng)
	}
	return weights
}

func (a *CausalAttention) CausalForward(x Matrix, train bool) Matrix {
	return MatMul(a.CausalWeights(x, train), MatMul(x, a.WValue))
}
