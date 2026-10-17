package gpt

// Step 14 (book 4.5) - the transformer block.
//
// One block is two sub-layers, each wrapped in the same recipe
// (normalize first, run the layer, add the input back):
//
//	x = x + attention(norm1(x))      <- tokens look at each other
//	x = x + feed_forward(norm2(x))   <- each token is processed alone
//
// The output has the same shape as the input, so blocks can be stacked as many
// times as you like. Normalizing BEFORE each layer is called pre-LayerNorm; it
// trains better than the original transformer's normalize-after. Dropout, which
// the book puts after each sub-layer, only runs in training, so it is left out.
//
// Tip: the attention here is the chapter 3 MultiHeadAttention unchanged.

import "llm-from-scratch/attention"

type TransformerBlock struct {
	Att          *attention.MultiHeadAttention
	FF           *FeedForward
	Norm1, Norm2 *LayerNorm
}

func NewTransformerBlock(cfg Config) *TransformerBlock {
	d := cfg.EmbDim
	att := &attention.MultiHeadAttention{
		WQuery: attention.NewMatrix(d, d), WKey: attention.NewMatrix(d, d), WValue: attention.NewMatrix(d, d),
		OutWeight: attention.NewMatrix(d, d), OutBias: make([]float32, d),
		NumHeads: cfg.NHeads, HeadDim: d / cfg.NHeads,
	}
	return &TransformerBlock{att, NewFeedForward(d), NewLayerNorm(d), NewLayerNorm(d)}
}

func addInPlace(a, b Matrix) Matrix {
	for i := range a.Data {
		a.Data[i] += b.Data[i]
	}
	return a
}

func (b *TransformerBlock) Forward(x Matrix) Matrix {
	x = addInPlace(b.Att.Forward(b.Norm1.Forward(x)), x) // add the original input back
	return addInPlace(b.FF.Forward(b.Norm2.Forward(x)), x)
}
