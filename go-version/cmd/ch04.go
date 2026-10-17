package main

// Chapter 4 demos: the GPT model, one building block at a time (book 4.1 to 4.7).
//
// Run from go-version/:  go run ./cmd ch04
//                        go run ./cmd ch04-124m    (the real 124M model)
//
// Tip: read the demos top to bottom. Numbers in 4.2, 4.3 and 4.4 match the book
// (the 4.4 weights are the book's torch.manual_seed(123) ones, written out so
// every language shares them). The model demos use random weights from the repo's
// shared generator, so all five languages print the same numbers; they differ from
// the book's torch output, and since the model is untrained its text is gibberish.

import (
	"fmt"
	"math"
	"strings"

	"llm-from-scratch/attention"
	"llm-from-scratch/gpt"
)

// Book 4.1: two short texts, already turned into token IDs by the GPT-2 tokenizer.
var batch = [][]int{{6109, 3626, 6100, 345}, {6109, 1110, 6622, 257}} // "Every effort moves you" / "Every day holds a"

// Book 4.2: the outputs of a Linear(5, 6) + ReLU layer for two inputs.
var layerOut = attention.FromRows([][]float32{
	{0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000},
	{0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000},
})

// Book 4.4: weights and biases of five Linear layers (3->3 four times, then 3->1), torch.manual_seed(123).
var shortcutLayers = []gpt.ExampleLayer{
	{W: [][]float64{{-0.23542964, 0.01912448, -0.28674594}, {0.21772662, -0.49193421, 0.42322308}, {-0.41964141, -0.45901766, -0.36482018}}, B: []float64{0.26147819, -0.21332639, 0.21605217}},
	{W: [][]float64{{-0.49001414, -0.35029206, -0.21198919}, {-0.11346072, -0.44043937, 0.37804362}, {-0.13615717, 0.18532233, 0.40826949}}, B: []float64{0.10756382, 0.15787685, 0.55729234}},
	{W: [][]float64{{-0.26039040, 0.18287641, -0.25687245}, {0.41260317, 0.46110451, -0.53230095}, {0.49285263, 0.27569306, 0.25159022}}, B: []float64{0.23768058, 0.47995073, -0.07623307}},
	{W: [][]float64{{-0.48826423, -0.16567004, -0.40661314}, {0.03816671, -0.10779562, -0.30968198}, {-0.05249349, 0.54698384, -0.04553875}}, B: []float64{0.01833092, -0.09004387, 0.09076351}},
	{W: [][]float64{{0.51442641, 0.35304558, 0.20493712}}, B: []float64{0.12546431}},
}

// fmt4 prints 4 decimals; a tiny negative number is just zero.
func fmt4(v float64) string {
	s := fmt.Sprintf("%.4f", v)
	if s == "-0.0000" {
		return "0.0000"
	}
	return s
}

func row4(values []float32) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmt4(float64(v))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func show4(label string, v []float32) { fmt.Printf("%s: %s\n", label, row4(v)) }

// commas writes 1234567 as "1,234,567".
func commas(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// commasF writes a float with two decimals and thousands separators.
func commasF(v float64) string {
	s := fmt.Sprintf("%.2f", v)
	dot := strings.Index(s, ".")
	return commas(atoi(s[:dot])) + s[dot:]
}

func atoi(s string) (n int) {
	fmt.Sscan(s, &n)
	return
}

func build(cfg gpt.Config) *gpt.Model { return gpt.NewInitializedModel(cfg, 123) }

// embed is token + position embedding for ids (what the model feeds its first block).
func embed(m *gpt.Model, ids []int) attention.Matrix {
	x := attention.NewMatrix(len(ids), m.Cfg.EmbDim)
	for t, id := range ids {
		for j := range x.Row(t) {
			x.Row(t)[j] = m.TokEmb.Row(id)[j] + m.PosEmb.Row(t)[j]
		}
	}
	return x
}

// demoBigPicture: 4.1, token IDs in, one score per vocabulary word out, for every position.
func demoBigPicture() {
	fmt.Println("== 4.1 the big picture (tiny GPT: emb 32, 4 heads, 2 blocks) ==")
	fmt.Printf("batch of token IDs, shape (%d, %d): %s\n", len(batch), len(batch[0]), "[[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]")
	model := build(gpt.ConfigTiny)
	logits := model.Forward(batch[0]) // the batch is processed one sequence at a time
	fmt.Printf("logits shape: (%d, %d, %d)  <- [batch, tokens, vocab_size]\n", len(batch), logits.Rows, logits.Cols)
	show4("logits[0][0], first 4 of 50257", logits.Row(0)[:4])
}

// demoLayerNorm: 4.2, make each token's numbers mean 0, variance 1.
func demoLayerNorm() {
	fmt.Println("== 4.2 layer normalization ==")
	means, vars := make([]float32, 2), make([]float32, 2)
	var norm0 []float32
	for i := 0; i < 2; i++ {
		row := layerOut.Row(i)
		var sum float32
		for _, v := range row {
			sum += v
		}
		means[i] = sum / float32(len(row))
		var sq float32
		for _, v := range row {
			sq += (v - means[i]) * (v - means[i])
		}
		vars[i] = sq / float32(len(row)-1) // the book first uses the unbiased variance
	}
	for _, v := range layerOut.Row(0) {
		norm0 = append(norm0, (v-means[0])/float32(math.Sqrt(float64(vars[0]))))
	}
	show4("mean per row", means)
	show4("variance per row", vars)
	show4("normalized, row 0", norm0)
	ln := gpt.NewLayerNorm(6).Forward(layerOut)
	outMean, outVar := make([]float32, 2), make([]float32, 2)
	for i := 0; i < 2; i++ {
		var sum float32
		for _, v := range ln.Row(i) {
			sum += v
		}
		outMean[i] = sum / 6
		var sq float32
		for _, v := range ln.Row(i) {
			sq += (v - outMean[i]) * (v - outMean[i])
		}
		outVar[i] = sq / 6
	}
	show4("LayerNorm output mean", outMean)
	show4("LayerNorm output variance (biased)", outVar)
}

// demoGeluAndFeedForward: 4.3, a smooth ReLU, and the expand-then-shrink feed forward block.
func demoGeluAndFeedForward() {
	fmt.Println("== 4.3 GELU and the feed forward network ==")
	x, g, r := make([]float32, 7), make([]float32, 7), make([]float32, 7)
	for i := range x {
		x[i] = float32(i - 3)
		g[i] = gpt.GELU(x[i])
		r[i] = float32(math.Max(0, float64(x[i])))
	}
	show4("x   ", x)
	show4("GELU", g)
	show4("ReLU", r)
	rng := attention.NewRng(123)
	ffn := gpt.NewFeedForward(32)
	ffn.Expand.W, ffn.Shrink.W = rng.Randn(32, 128, 0.02), rng.Randn(128, 32, 0.02)
	out := ffn.Forward(rng.Rand(3, 32))
	fmt.Printf("feed forward: input (2, 3, 32) -> output (2, %d, %d) (hidden layer is 4 x 32 = 128 wide)\n", out.Rows, out.Cols)
}

// demoShortcut: 4.4, shortcut connections keep the gradient alive.
func demoShortcut() {
	fmt.Println("== 4.4 shortcut connections (mean |gradient| of each layer's weights) ==")
	for _, useShortcut := range []bool{false, true} {
		net := &gpt.ExampleDeepNeuralNetwork{Layers: shortcutLayers, UseShortcut: useShortcut}
		grads := net.GradientMeans([]float64{1, 0, -1}, 0)
		parts := make([]string, len(grads))
		for i, g := range grads {
			parts[i] = fmt.Sprintf("%.6f", g)
		}
		label := "without"
		if useShortcut {
			label = "with   "
		}
		fmt.Printf("%s shortcut: %s   (layer 0 ... layer 4)\n", label, strings.Join(parts, "  "))
	}
}

// demoTransformerBlock: 4.5, attention + feed forward, each wrapped in norm and shortcut. Shape in = shape out.
func demoTransformerBlock() {
	fmt.Println("== 4.5 the transformer block ==")
	model := build(gpt.ConfigTiny)
	x := embed(model, batch[0])
	out := model.Blocks[0].Forward(x)
	fmt.Printf("input shape: (%d, %d, %d)  output shape: (%d, %d, %d)\n", len(batch), x.Rows, x.Cols, len(batch), out.Rows, out.Cols)
	show4("input[0][0], first 4 numbers", x.Row(0)[:4])
	show4("output[0][0], first 4 numbers", out.Row(0)[:4])
}

// demoParameters: 4.6, how big is the model?
func demoParameters() {
	fmt.Println("== 4.6 model size ==")
	tiny := build(gpt.ConfigTiny)
	fmt.Printf("tiny model: %s parameters (arithmetic says %s)\n", commas(tiny.CountParameters()), commas(gpt.ParameterCount(gpt.ConfigTiny)))
	cfg := gpt.Config124M
	total := gpt.ParameterCount(cfg)
	head := cfg.VocabSize * cfg.EmbDim
	mb := func(n int) float64 { return float64(n) * 4 / (1024 * 1024) }
	fmt.Printf("GPT-2 small, no model built: %s parameters\n", commas(total))
	fmt.Printf("  minus the output head (weight tying in the original GPT-2): %s\n", commas(total-head))
	fmt.Printf("  float32 size: %.2f MB\n", mb(total))
	d := cfg.EmbDim
	ff := d*4*d + 4*d + 4*d*d + d
	att := 4*d*d + d
	fmt.Printf("exercise 4.1: feed forward %s vs attention %s parameters per block\n", commas(ff), commas(att))
	fmt.Println("exercise 4.2: bigger models, same code, different configuration")
	fmt.Printf("  %-20s %14s %14s %11s\n", "", "parameters", "weight-tied", "float32 MB")
	for _, s := range gpt.Sizes {
		c := cfg
		c.EmbDim, c.NLayers, c.NHeads = s.EmbDim, s.NLayers, s.NHeads
		n := gpt.ParameterCount(c)
		fmt.Printf("  %-20s %14s %14s %11s\n", s.Name, commas(n), commas(n-c.VocabSize*c.EmbDim), commasF(mb(n)))
	}
}

// demoGenerate: 4.7, predict one token, append it, repeat.
func demoGenerate(cfg gpt.Config, label string) {
	fmt.Printf("== 4.7 generating text (%s) ==\n", label)
	model := build(cfg)
	enc := newBPE()
	encoded := enc.Encode("Hello, I am", nil, nil)
	fmt.Printf("encoded: %s\n", formatIDs(encoded))
	out := gpt.GenerateTextSimple(model, encoded, 6, cfg.ContextLength)
	fmt.Printf("output IDs: %s\n", formatIDs(out))
	fmt.Printf("output length: %d\n", len(out))
	fmt.Printf("decoded: %s\n", enc.Decode(out))
	fmt.Println("(gibberish: the weights are random, nothing has been trained yet)")
}

func runChapter4() {
	demoBigPicture()
	demoLayerNorm()
	demoGeluAndFeedForward()
	demoShortcut()
	demoTransformerBlock()
	demoParameters()
	demoGenerate(gpt.ConfigTiny, "tiny model")
}

// run124M is the real 124M-parameter model: forward pass, then generation. Needs ~1 GB of memory.
func run124M() {
	model := build(gpt.Config124M)
	fmt.Println("== GPT-2 small (124M) ==")
	fmt.Printf("parameters built: %s\n", commas(model.CountParameters()))
	logits := model.Forward(batch[0])
	fmt.Printf("logits shape: (%d, %d, %d)\n", len(batch), logits.Rows, logits.Cols)
	show4("logits[0][0], first 4 of 50257", logits.Row(0)[:4])
	demoGenerate(gpt.Config124M, "124M model")
}
