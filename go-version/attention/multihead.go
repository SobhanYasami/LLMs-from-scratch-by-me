package attention

// Step 9 (book 3.6) - multi-head attention.
//
// One attention head learns one way of relating tokens. Several heads, each
// with its own smaller query/key/value, can look for different patterns in
// parallel (one for grammar, one for topic, ...). Their outputs are joined
// and mixed by a final linear layer.
//
//	d_out = 4, num_heads = 2  ->  each head works on head_dim = 2 columns
//
//	W_query columns: [ head 0 | head 1 ]     (same for W_key and W_value)
//	per head: causal attention on its own slice -> [T, head_dim]
//	concatenate heads -> [T, d_out] -> out_proj
//
// Tip: one big weight matrix cut into column slices is the same as separate
// small matrices per head, just faster (one matmul instead of many).

type MultiHeadAttention struct {
	WQuery, WKey, WValue Matrix
	OutWeight            Matrix
	OutBias              []float32
	NumHeads, HeadDim    int
}

// NewRandomMultiHead draws uniform [0, 1) weights (query, key, value,
// out_proj) and a zero bias. dOut must divide evenly by numHeads.
func NewRandomMultiHead(dIn, dOut, numHeads int, rng *Rng) *MultiHeadAttention {
	wq, wk, wv := rng.Rand(dIn, dOut), rng.Rand(dIn, dOut), rng.Rand(dIn, dOut)
	return &MultiHeadAttention{wq, wk, wv, rng.Rand(dOut, dOut), make([]float32, dOut), numHeads, dOut / numHeads}
}

// Forward takes x [T, d_in] and returns [T, d_out].
func (m *MultiHeadAttention) Forward(x Matrix) Matrix {
	q, k, v := MatMul(x, m.WQuery), MatMul(x, m.WKey), MatMul(x, m.WValue)
	context := NewMatrix(x.Rows, m.NumHeads*m.HeadDim)
	for h := 0; h < m.NumHeads; h++ {
		lo, hi := h*m.HeadDim, (h+1)*m.HeadDim // this head's column slice
		qh, kh, vh := SliceCols(q, lo, hi), SliceCols(k, lo, hi), SliceCols(v, lo, hi)
		scores := MaskFuture(MatMul(qh, Transpose(kh)))
		weights := SoftmaxRows(scaleBySqrt(scores, m.HeadDim))
		head := MatMul(weights, vh)
		for i := 0; i < head.Rows; i++ { // glue the heads back together
			copy(context.Row(i)[lo:hi], head.Row(i))
		}
	}
	out := MatMul(context, m.OutWeight)
	for i := 0; i < out.Rows; i++ {
		for j := 0; j < out.Cols; j++ {
			out.Set(i, j, out.At(i, j)+m.OutBias[j])
		}
	}
	return out
}
