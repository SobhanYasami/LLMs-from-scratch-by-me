// Package attention holds the attention building blocks (book chapter 3).
//
// Files, in the order you learn them:
//   - matrix.go:         the few matrix operations attention needs.
//   - rng.go:            a seeded random generator shared with the other languages.
//   - simple.go:         simplified self-attention, no trainable weights (3.3).
//   - self_attention.go: scaled dot-product attention with W_query/key/value (3.4).
//   - causal.go:         hide future tokens with a mask, optional dropout (3.5).
//   - multihead.go:      several heads in parallel plus an output projection (3.6).
//
// Tip: there is no tensor library here, so Matrix is a row-major []float32.
// Everything attention does is these operations: matmul, transpose, softmax.
package attention

import "math"

// Matrix is a row-major rows x cols matrix of float32.
type Matrix struct {
	Rows, Cols int
	Data       []float32
}

func NewMatrix(rows, cols int) Matrix {
	return Matrix{rows, cols, make([]float32, rows*cols)}
}

// FromRows builds a matrix from a slice of equal-length rows.
func FromRows(rows [][]float32) Matrix {
	m := NewMatrix(len(rows), len(rows[0]))
	for i, r := range rows {
		copy(m.Data[i*m.Cols:], r)
	}
	return m
}

func (m Matrix) At(i, j int) float32     { return m.Data[i*m.Cols+j] }
func (m Matrix) Set(i, j int, v float32) { m.Data[i*m.Cols+j] = v }
func (m Matrix) Row(i int) []float32     { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// MatMul returns a x b ([n,k] x [k,m] -> [n,m]).
// Tip: the loops run i, k, j so both matrices are read row by row, which is
// much faster on big matrices. Each output element is still summed over k in
// order, so the numbers are the same as with the textbook i, j, k order.
func MatMul(a, b Matrix) Matrix {
	out := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		outRow := out.Row(i)
		for k := 0; k < a.Cols; k++ {
			aik := a.At(i, k)
			bRow := b.Row(k)
			for j, bkj := range bRow {
				outRow[j] += aik * bkj
			}
		}
	}
	return out
}

func Transpose(m Matrix) Matrix {
	out := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out.Set(j, i, m.At(i, j))
		}
	}
	return out
}

// SoftmaxRows turns every row into weights that are positive and sum to 1.
// Tip: subtracting the row maximum first does not change the result but keeps
// exp() from overflowing. A -Inf entry becomes exactly 0.
func SoftmaxRows(m Matrix) Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		row := m.Row(i)
		max := row[0]
		for _, v := range row {
			if v > max {
				max = v
			}
		}
		var sum float32
		for j, v := range row {
			e := float32(math.Exp(float64(v - max)))
			out.Set(i, j, e)
			sum += e
		}
		for j := 0; j < m.Cols; j++ {
			out.Set(i, j, out.At(i, j)/sum)
		}
	}
	return out
}

// Scale multiplies every element by s.
func Scale(m Matrix, s float32) Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	for i, v := range m.Data {
		out.Data[i] = v * s
	}
	return out
}

// MaskFuture sets every element above the diagonal to -Inf (the "future" tokens).
func MaskFuture(m Matrix) Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	copy(out.Data, m.Data)
	for i := 0; i < m.Rows; i++ {
		for j := i + 1; j < m.Cols; j++ {
			out.Set(i, j, float32(math.Inf(-1)))
		}
	}
	return out
}

// SliceCols returns columns [start, end) as a new matrix.
func SliceCols(m Matrix, start, end int) Matrix {
	out := NewMatrix(m.Rows, end-start)
	for i := 0; i < m.Rows; i++ {
		copy(out.Row(i), m.Row(i)[start:end])
	}
	return out
}
