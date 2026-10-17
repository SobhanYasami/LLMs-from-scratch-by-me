package main

// Chapter 5 demos: pretraining, one step at a time (book 5.1 to 5.5).
//
// Run from go-version/:  go run ./cmd ch05
//                        go run ./cmd ch05-gpt2    (OpenAI's weights, 5.5)
//
// Tip: read the demos top to bottom. The model is the tiny GPT of chapter 4 (emb 32, 2 blocks,
// context 16) so that every language can TRAIN it in seconds; the book trains the 124M model on a
// GPU. All five languages use the same weights, data order (no shuffling) and sampling generator,
// so they print the same numbers. They differ from the book's torch output.

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/pkoukk/tiktoken-go"

	"llm-from-scratch/attention"
	"llm-from-scratch/gpt"
	"llm-from-scratch/tokenization"
	"llm-from-scratch/train"
)

const (
	gpt2Path    = "../assets/gpt2/model.safetensors"
	batchSize   = 4
	stride      = 16
	epochs      = 2
	evalFreq    = 24
	evalIter    = 2
	lr          = 3e-3
	weightDecay = 0.1
)

// cfg5 is the tiny GPT with dropout off: our loop has no random masks.
var cfg5 = func() gpt.Config { c := gpt.ConfigTiny; c.DropRate = 0; return c }()

func fmtD(v float32, digits int) string {
	s := fmt.Sprintf("%.*f", digits, v)
	if strings.HasPrefix(s, "-") && strings.Trim(s[1:], "0.") == "" {
		return s[1:]
	}
	return s
}

func rowD(values []float32, digits int) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmtD(v, digits)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// quoted is text in single quotes with newlines flattened (same in every language).
func quoted(text string) string { return "'" + strings.ReplaceAll(text, "\n", " ") + "'" }

func build5() *gpt.Model { return gpt.NewInitializedModel(cfg5, 123) }

// loaders cuts the story 90% / 10% into windows and batches them in book order.
func loaders(enc *tiktoken.Tiktoken, ctx, batch int) (trainB, valB []tokenization.Batch) {
	text := loadText()
	split := int(0.9 * float64(len(text)))
	make := func(t string, dropLast bool) []tokenization.Batch {
		ds := tokenization.NewGPTDataset(enc.Encode(t, nil, nil), ctx, stride)
		return tokenization.NewDataLoader(ds, batch, false, dropLast, 123).Epoch()
	}
	return make(text[:split], true), make(text[split:], false)
}

// demoLoss is 5.1: from probabilities to a single number.
func demoLoss(enc *tiktoken.Tiktoken) {
	fmt.Println("== 5.1 evaluating generative text models ==")
	model := build5()
	inputs := [][]int{{16833, 3626, 6100}, {40, 1107, 588}}  // ["every effort moves", "I really like"]
	targets := [][]int{{3626, 6100, 345}, {588, 428, 11311}} // [" effort moves you", " really like chocolate"]
	fmt.Println("inputs : [[16833, 3626, 6100], [40, 1107, 588]]")
	fmt.Println("targets: [[3626, 6100, 345], [588, 428, 11311]]")
	fmt.Printf("probas shape: (2, 3, %d)  <- [batch, tokens, vocab_size]\n", cfg5.VocabSize)
	var logProbas []float32
	var ceTotal float32
	for i := range inputs {
		logits := model.Forward(inputs[i])
		ceTotal += train.CrossEntropySum(logits, targets[i])
		picked := make([]float32, 3)
		for t := 0; t < 3; t++ {
			picked[t] = train.Softmax(logits.Row(t))[targets[i][t]]
			logProbas = append(logProbas, float32(math.Log(float64(picked[t]))))
		}
		fmt.Printf("probability of the target, text %d: %s\n", i+1, rowD(picked, 6))
	}
	fmt.Printf("log probabilities: %s\n", rowD(logProbas, 4))
	var sum float32
	for _, v := range logProbas {
		sum += v
	}
	avg := sum / float32(len(logProbas))
	fmt.Printf("average log probability: %s\n", fmtD(avg, 4))
	fmt.Printf("negative average (the loss): %s\n", fmtD(-avg, 4))
	loss := ceTotal / 6
	fmt.Printf("cross_entropy of the whole batch at once: %s\n", fmtD(loss, 4))
	fmt.Printf("perplexity: %s  (a model that guesses uniformly is torn between 50,257 tokens)\n", fmtD(train.Perplexity(loss), 0))
	trainB, valB := loaders(enc, cfg5.ContextLength, batchSize)
	fmt.Printf("training batches: %d of %d x %d tokens, validation batches: %d\n", len(trainB), batchSize, cfg5.ContextLength, len(valB))
	tl, vl := train.CalcLossLoader(trainB, model, 0), train.CalcLossLoader(valB, model, 0)
	fmt.Printf("untrained: training loss %s, validation loss %s  (a uniform guess gives ln(50257) = %s)\n", fmtD(tl, 3), fmtD(vl, 3), fmtD(float32(math.Log(50257)), 3))
}

func generateAndPrintSample(m *gpt.Model, enc *tiktoken.Tiktoken, start string) {
	ids := enc.Encode(start, []string{"<|endoftext|>"}, nil)
	out := gpt.GenerateTextSimple(m, ids, 20, m.Cfg.ContextLength)
	fmt.Println(strings.ReplaceAll(enc.Decode(out), "\n", " "))
}

// trainStep is one training step: zero the old gradients, forward and backward, update the weights.
func trainStep(m, grads *gpt.Model, opt *train.AdamW, b tokenization.Batch) float32 {
	train.ZeroGrads(grads)
	loss := train.LossAndGrads(m, grads, b)
	opt.StepWith(m, grads)
	return loss
}

// trainDemo is 5.2: train for a few epochs. Returns the trained model, its optimizer and the gradient buffer.
func trainDemo(enc *tiktoken.Tiktoken) (*gpt.Model, *train.AdamW, *gpt.Model) {
	fmt.Println("== 5.2 training an LLM ==")
	m := build5()
	grads := m.ZerosLike()
	opt := train.NewAdamW(m, lr, weightDecay)
	trainB, valB := loaders(enc, cfg5.ContextLength, batchSize)
	step := 0
	for epoch := 0; epoch < epochs; epoch++ {
		for _, b := range trainB {
			trainStep(m, grads, opt, b)
			if step%evalFreq == 0 {
				tl, vl := train.CalcLossLoader(trainB, m, evalIter), train.CalcLossLoader(valB, m, evalIter)
				fmt.Printf("Ep %d (Step %06d): Train loss %s, Val loss %s\n", epoch+1, step, fmtD(tl, 3), fmtD(vl, 3))
			}
			step++
		}
		generateAndPrintSample(m, enc, "Every effort moves you")
	}
	return m, opt, grads
}

var vocab5 = []string{"closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you"}
var nextTokenLogits = []float32{4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79}

// demoDecoding is 5.3: temperature and top-k.
func demoDecoding(m *gpt.Model, enc *tiktoken.Tiktoken) {
	fmt.Println("== 5.3 decoding strategies ==")
	probas := train.Softmax(nextTokenLogits)
	fmt.Printf("next-token probabilities: %s\n", rowD(probas, 4))
	greedy := 0
	for i, p := range probas {
		if p > probas[greedy] {
			greedy = i
		}
	}
	fmt.Printf("greedy pick: %s\n", vocab5[greedy])
	for _, t := range []struct {
		label string
		value float32
	}{{"1", 1}, {"0.1", 0.1}, {"5", 5}} {
		p := train.SoftmaxWithTemperature(nextTokenLogits, t.value)
		rng := attention.NewRng(123)
		counts := make([]int, len(vocab5))
		for i := 0; i < 1000; i++ {
			counts[train.SampleIndex(p, rng)]++
		}
		var shown []string
		for i, c := range counts {
			if c > 0 {
				shown = append(shown, fmt.Sprintf("%d x %s", c, vocab5[i]))
			}
		}
		fmt.Printf("temperature %s: P(pizza) = %s  1000 draws: %s\n", t.label, fmtD(p[6], 4), strings.Join(shown, ", "))
	}
	order := []int{0, 1, 2, 3, 4, 5, 6, 7, 8}
	sort.SliceStable(order, func(a, b int) bool { return nextTokenLogits[order[a]] > nextTokenLogits[order[b]] })
	top := []float32{nextTokenLogits[order[0]], nextTokenLogits[order[1]], nextTokenLogits[order[2]]}
	fmt.Printf("top-3 logits: %s at positions %s\n", rowD(top, 2), formatIDs(order[:3]))
	masked := train.TopKLogits(nextTokenLogits, 3)
	shown := make([]string, len(masked))
	for i, v := range masked {
		if math.IsInf(float64(v), -1) {
			shown[i] = "'-inf'"
		} else {
			shown[i] = "'" + fmtD(v, 2) + "'"
		}
	}
	fmt.Printf("after masking with -inf: [%s]\n", strings.Join(shown, ", "))
	fmt.Printf("top-3 probabilities: %s\n", rowD(train.Softmax(masked), 4))
	start := enc.Encode("Every effort moves you", []string{"<|endoftext|>"}, nil)
	ctx := m.Cfg.ContextLength
	greedyIDs := gpt.GenerateTextSimple(m, start, 15, ctx)
	fmt.Printf("greedy   : %s\n", quoted(enc.Decode(greedyIDs)))
	sampled := train.Generate(m, start, 15, ctx, attention.NewRng(123), 1.4, 25)
	fmt.Printf("top-k 25, temperature 1.4: %s\n", quoted(enc.Decode(sampled)))
	same := train.Generate(m, start, 15, ctx, attention.NewRng(123), 0, 0)
	fmt.Printf("temperature 0 equals greedy: %t\n", fmt.Sprint(same) == fmt.Sprint(greedyIDs))
}

// demoCheckpoint is 5.4: save, load, and carry on exactly where we stopped.
func demoCheckpoint(m *gpt.Model, opt *train.AdamW, grads *gpt.Model, enc *tiktoken.Tiktoken) {
	fmt.Println("== 5.4 saving and loading ==")
	path := "../assets/checkpoints/go.bin"
	check(train.SaveCheckpoint(path, m, opt))
	info, err := os.Stat(path)
	check(err)
	fmt.Printf("saved model and optimizer state: %s bytes\n", commas(int(info.Size())))
	restored, restoredOpt, err := train.LoadCheckpoint(path, lr, weightDecay)
	check(err)
	x := []int{6109, 3626, 6100, 345}
	a, b := m.Forward(x), restored.Forward(x)
	var diff float32
	for i := range a.Data {
		diff = max(diff, float32(math.Abs(float64(a.Data[i]-b.Data[i]))))
	}
	fmt.Printf("largest logit difference between the two models: %s\n", fmtD(diff, 6))
	trainB, _ := loaders(enc, cfg5.ContextLength, batchSize)
	restoredGrads := restored.ZerosLike()
	var continued, resumed []float32
	for _, batch := range trainB[:3] {
		continued = append(continued, trainStep(m, grads, opt, batch))
	}
	for _, batch := range trainB[:3] {
		resumed = append(resumed, trainStep(restored, restoredGrads, restoredOpt, batch))
	}
	fmt.Printf("3 more steps, continued: losses %s\n", rowD(continued, 3))
	fmt.Printf("3 more steps, restored : losses %s\n", rowD(resumed, 3))
	fmt.Printf("same losses: %t\n", fmt.Sprint(continued) == fmt.Sprint(resumed))
}

func runChapter5() {
	enc := newBPE()
	demoLoss(enc)
	m, opt, grads := trainDemo(enc)
	demoDecoding(m, enc)
	demoCheckpoint(m, opt, grads, enc)
}

// runGPT2 is 5.5: OpenAI's GPT-2 small, loaded into our model. Needs assets/gpt2/model.safetensors.
func runGPT2() {
	enc := newBPE()
	if _, err := os.Stat(gpt2Path); err != nil {
		fmt.Fprintf(os.Stderr, "missing %s: download it first (command in train/openai.go)\n", gpt2Path)
		os.Exit(1)
	}
	fmt.Println("== 5.5 loading pretrained weights from OpenAI ==")
	m, err := train.LoadGPT2Weights(gpt2Path)
	check(err)
	fmt.Printf("loaded GPT-2 small: %s parameters (with the attention biases)\n", commas(m.CountParameters()))
	start := enc.Encode("Every effort moves you", []string{"<|endoftext|>"}, nil)
	ctx := train.GPT2Small.ContextLength
	out := train.Generate(m, start, 25, ctx, attention.NewRng(123), 1.5, 50)
	fmt.Printf("generated: %s\n", quoted(enc.Decode(out)))
	greedy := gpt.GenerateTextSimple(m, start, 25, ctx)
	fmt.Printf("greedy   : %s\n", quoted(enc.Decode(greedy)))
	text := loadText()
	split := int(0.9 * float64(len(text)))
	ds := tokenization.NewGPTDataset(enc.Encode(text[split:], nil, nil), 64, 64)
	val := tokenization.NewDataLoader(ds, 2, false, true, 123).Epoch()
	fmt.Printf("exercise 5.5: loss of the pretrained model on The Verdict (validation part, 2 batches of 2 x 64 tokens): %s\n", fmtD(train.CalcLossLoader(val, m, 2), 3))
}
