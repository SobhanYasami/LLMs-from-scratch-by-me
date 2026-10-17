// Package gpt holds the GPT model (book chapter 4).
//
// Files, in the order you learn them:
//   - layers.go:   layer normalization, GELU, Linear and the feed forward block (4.2, 4.3).
//   - shortcut.go: a tiny deep network that shows what shortcut connections do (4.4).
//   - block.go:    the transformer block (4.5).
//   - model.go:    the GPT model, its configuration and text generation (4.6, 4.7).
//
// Tip: like chapter 3 there is no tensor library. A sequence of T tokens is a
// Matrix [T, D]; the Go code handles one sequence at a time (the demo loops
// over the batch), and the math per sequence is the same as the batched Python.
package gpt

// Steps 10-12 (book 4.2 and 4.3) - layer normalization, GELU and the feed forward block.
//
// Layer normalization rescales each token's vector to mean 0 and variance 1,
// which keeps activations in a range where training is stable:
//
//	x [T, D]  ->  (x - mean) / sqrt(var + eps)  ->  scale * that + shift
//
// The mean and variance are taken over the D numbers of ONE token (one row),
// never across tokens. scale and shift are learned, so the model can undo the
// normalization where that helps.
//
// GELU is a smooth cousin of ReLU, and the feed forward block expands each
// token's vector 4x, applies GELU, and shrinks it back:
//
//	[T, D] -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> [T, D]

import (
	"math"

	"llm-from-scratch/attention"
)

type Matrix = attention.Matrix

type LayerNorm struct {
	Scale, Shift []float32
}

func NewLayerNorm(dim int) *LayerNorm {
	l := &LayerNorm{make([]float32, dim), make([]float32, dim)}
	for i := range l.Scale {
		l.Scale[i] = 1
	}
	return l
}

const layerNormEps = 1e-5

// Forward normalizes every row of x.
func (l *LayerNorm) Forward(x Matrix) Matrix {
	out := attention.NewMatrix(x.Rows, x.Cols)
	n := float32(x.Cols)
	for i := 0; i < x.Rows; i++ {
		row := x.Row(i)
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= n
		var variance float32 // divide by n (biased), like GPT-2
		for _, v := range row {
			variance += (v - mean) * (v - mean)
		}
		variance /= n
		inv := float32(1 / math.Sqrt(float64(variance+layerNormEps)))
		for j, v := range row {
			out.Set(i, j, l.Scale[j]*((v-mean)*inv)+l.Shift[j])
		}
	}
	return out
}

// GELU is the tanh approximation GPT-2 was trained with.
func GELU(x float32) float32 {
	v := float64(x)
	return float32(0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v))))
}

// Linear is x @ W + b with W stored [in, out] (PyTorch's nn.Linear stores it transposed).
type Linear struct {
	W Matrix
	B []float32
}

func NewLinear(in, out int) *Linear {
	return &Linear{attention.NewMatrix(in, out), make([]float32, out)}
}

func (l *Linear) Forward(x Matrix) Matrix {
	out := attention.MatMul(x, l.W)
	for i := 0; i < out.Rows; i++ {
		row := out.Row(i)
		for j := range row {
			row[j] += l.B[j]
		}
	}
	return out
}

type FeedForward struct{ Expand, Shrink *Linear }

func NewFeedForward(dim int) *FeedForward {
	return &FeedForward{NewLinear(dim, 4*dim), NewLinear(4*dim, dim)}
}

func (f *FeedForward) Forward(x Matrix) Matrix {
	h := f.Expand.Forward(x)
	for i, v := range h.Data {
		h.Data[i] = GELU(v)
	}
	return f.Shrink.Forward(h)
}
