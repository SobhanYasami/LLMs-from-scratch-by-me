package main

// Chapter 3 demos: attention, one idea at a time (book 3.3 to 3.6).
//
// Run from go-version/:  go run ./cmd ch03
//
// Tip: read the demos top to bottom; each builds on the one before. The numbers
// in 3.3 and 3.4 match the book exactly (3.4 uses the book's seed-123 weights,
// hard-coded so every language shares them).

import (
	"fmt"
	"math"
	"strings"

	"llm-from-scratch/attention"
)

// "Your journey starts with one step." as 3-dimensional embeddings (book 3.3.1).
var inputs = attention.FromRows([][]float32{
	{0.43, 0.15, 0.89}, // Your
	{0.55, 0.87, 0.66}, // journey
	{0.57, 0.85, 0.64}, // starts
	{0.22, 0.58, 0.33}, // with
	{0.77, 0.25, 0.10}, // one
	{0.05, 0.80, 0.55}, // step
})

// The book's torch.manual_seed(123) weights (d_in=3, d_out=2), written out so no language needs torch.
var (
	wQuery = attention.FromRows([][]float32{{0.29611194, 0.51656228}, {0.25167072, 0.68855679}, {0.07397246, 0.86652195}})
	wKey   = attention.FromRows([][]float32{{0.13657987, 0.10247904}, {0.18405646, 0.72644675}, {0.31525391, 0.68710667}})
	wValue = attention.FromRows([][]float32{{0.07563531, 0.19663817}, {0.31641197, 0.40174013}, {0.11856830, 0.82739538}})
)

func fmtRow(values []float32) string {
	parts := make([]string, len(values))
	for i, v := range values {
		if math.IsInf(float64(v), -1) {
			parts[i] = "-inf"
		} else {
			parts[i] = fmt.Sprintf("%.4f", v)
		}
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func showVec(label string, v []float32) { fmt.Printf("%s: %s\n", label, fmtRow(v)) }

func showMatrix(label string, m attention.Matrix) {
	fmt.Printf("%s:\n", label)
	for i := 0; i < m.Rows; i++ {
		fmt.Println("  " + fmtRow(m.Row(i)))
	}
}

// demoSimpleAttention: 3.3, one query by hand, then all tokens at once.
func demoSimpleAttention() {
	fmt.Println("== 3.3.1 simple attention for one query (x2, 'journey') ==")
	query := attention.FromRows([][]float32{inputs.Row(1)})
	scores2 := attention.MatMul(query, attention.Transpose(inputs)) // dot product of every token with the query
	weights2 := attention.SoftmaxRows(scores2)
	context2 := attention.MatMul(weights2, inputs)
	showVec("scores    ", scores2.Row(0))
	showVec("weights   ", weights2.Row(0))
	showVec("context z2", context2.Row(0))

	fmt.Println("== 3.3.2 all tokens at once ==")
	scores, weights, context := attention.SimpleAttention(inputs)
	showMatrix("scores (T x T)", scores)
	showMatrix("weights (rows sum to 1)", weights)
	sums := make([]float32, weights.Rows)
	for i := range sums {
		for _, w := range weights.Row(i) {
			sums[i] += w
		}
	}
	showVec("row sums", sums)
	showMatrix("context vectors (T x d)", context)
}

// demoSelfAttention: 3.4, query/key/value with trainable (here: fixed) weights.
func demoSelfAttention() {
	fmt.Println("== 3.4 self-attention with trainable weights (d_in=3, d_out=2) ==")
	sa := &attention.SelfAttention{WQuery: wQuery, WKey: wKey, WValue: wValue}
	x2 := attention.FromRows([][]float32{inputs.Row(1)})
	query2 := attention.MatMul(x2, wQuery)
	keys, values := attention.MatMul(inputs, wKey), attention.MatMul(inputs, wValue)
	showVec("query_2", query2.Row(0))
	fmt.Printf("keys shape: (%d, %d), values shape: (%d, %d)\n", keys.Rows, keys.Cols, values.Rows, values.Cols)
	scores2 := attention.MatMul(query2, attention.Transpose(keys))
	showVec("scores_2 (q2 . keys)", scores2.Row(0))
	scaled := attention.Scale(scores2, 1/float32(math.Sqrt(float64(keys.Cols))))
	weights2 := attention.SoftmaxRows(scaled)
	showVec("weights_2 (scaled)", weights2.Row(0))
	showVec("context_2", attention.MatMul(weights2, values).Row(0))
	showMatrix("all context vectors (T x d_out)", sa.Forward(inputs))
}

// demoCausalAttention: 3.5, mask the future, then dropout.
func demoCausalAttention() {
	fmt.Println("== 3.5 causal attention ==")
	ca := attention.NewCausalAttention(wQuery, wKey, wValue, 0, nil)
	showMatrix("causal weights (upper triangle is 0)", ca.CausalWeights(inputs, false))
	showMatrix("causal context vectors", ca.CausalForward(inputs, false))
	ones := attention.NewMatrix(6, 6)
	for i := range ones.Data {
		ones.Data[i] = 1
	}
	dropped := attention.Dropout(ones, 0.5, attention.NewRng(123))
	showMatrix("dropout 0.5 on ones(6, 6), seed 123 (kept values are doubled)", dropped)
}

// demoMultiHead: 3.6, split the work across heads.
func demoMultiHead() {
	fmt.Println("== 3.6 multi-head attention (d_in=3, d_out=2, heads=2) ==")
	mha := attention.NewRandomMultiHead(3, 2, 2, attention.NewRng(123))
	// A batch is just the same computation repeated per example; two copies = a batch of 2.
	out := mha.Forward(inputs)
	fmt.Printf("input shape: (2, %d, %d)  output shape: (2, %d, %d)\n", inputs.Rows, inputs.Cols, out.Rows, out.Cols)
	showMatrix("output[0]", out)
}

func runChapter3() {
	demoSimpleAttention()
	demoSelfAttention()
	demoCausalAttention()
	demoMultiHead()
}
