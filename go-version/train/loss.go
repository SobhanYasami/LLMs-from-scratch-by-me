// Package train holds pretraining (book chapter 5).
//
// Files, in the order you learn them:
//   - loss.go:       cross entropy loss and perplexity (5.1).
//   - grad.go:       the backward pass, written by hand, and adamw.go, the optimizer (5.2).
//   - sample.go:     temperature and top-k decoding (5.3).
//   - checkpoint.go: saving and loading weights and optimizer state (5.4).
//   - openai.go:     loading OpenAI's pretrained GPT-2 weights (5.5).
//
// The Python version lets torch's autograd do the backward pass. Here there is no autograd, so
// grad.go applies the chain rule layer by layer.
package train

// Step 17 (book 5.1) - how wrong is the model? Cross entropy loss and perplexity.
//
// For every position the model gives a score (logit) for each of the 50,257 tokens. Softmax turns
// the scores into probabilities. The loss looks only at the probability given to the token that
// REALLY came next:
//
//	loss = - mean( log p(target) )        (cross entropy)
//	perplexity = exp(loss)                "among how many tokens is the model torn?"
//
// A model that guesses uniformly has loss ln(50257) = 10.82. A perfect model has loss 0.

import (
	"math"

	"llm-from-scratch/attention"
	"llm-from-scratch/gpt"
	"llm-from-scratch/tokenization"
)

// LogSumExp is the log of softmax's denominator: log(sum(exp(x - max))) + max.
func LogSumExp(row []float32) float32 {
	max := float32(math.Inf(-1))
	for _, v := range row {
		if v > max {
			max = v
		}
	}
	var sum float32
	for _, v := range row {
		sum += float32(math.Exp(float64(v - max)))
	}
	return float32(math.Log(float64(sum))) + max
}

// CrossEntropySum is the cross entropy of logits [T, vocab] against the targets (one token ID per row), summed over rows.
func CrossEntropySum(logits attention.Matrix, targets []int) float32 {
	var total float32
	for i, t := range targets {
		row := logits.Row(i)
		total += LogSumExp(row) - row[t] // -log p(target)
	}
	return total
}

func Perplexity(loss float32) float32 { return float32(math.Exp(float64(loss))) }

// CalcLossBatch is the mean cross entropy of one batch (all its tokens count equally).
func CalcLossBatch(b tokenization.Batch, m *gpt.Model) float32 {
	var total float32
	n := 0
	for i, x := range b.Inputs {
		total += CrossEntropySum(m.Forward(x), b.Targets[i])
		n += len(b.Targets[i])
	}
	return total / float32(n)
}

// CalcLossLoader is the average loss over (at most numBatches; 0 means all) batches.
func CalcLossLoader(batches []tokenization.Batch, m *gpt.Model, numBatches int) float32 {
	if len(batches) == 0 {
		return float32(math.NaN())
	}
	n := len(batches)
	if numBatches > 0 && numBatches < n {
		n = numBatches
	}
	var total float32
	for _, b := range batches[:n] {
		total += CalcLossBatch(b, m)
	}
	return total / float32(n)
}
