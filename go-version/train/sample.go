package train

// Step 19 (book 5.3) - decoding strategies: temperature and top-k sampling.
//
// Greedy decoding always takes the most likely token, so the same prompt gives the same text,
// often copied straight from the training data. Two knobs add variety:
//
//   - temperature: divide the logits by T before softmax. T < 1 sharpens the distribution (more
//     confident), T > 1 flattens it (more surprising).
//   - top-k: keep only the k best logits, set the others to -inf (probability 0).
//
// Then draw the next token IN PROPORTION to its probability. The draw uses the repo's shared
// generator (not torch.multinomial), so every language agrees.

import (
	"math"
	"sort"

	"llm-from-scratch/attention"
	"llm-from-scratch/gpt"
)

// Softmax of one row of scores, like torch.softmax: subtract the max, exp, divide by the sum.
func Softmax(logits []float32) []float32 {
	max := float32(math.Inf(-1))
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	exps := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		exps[i] = float32(math.Exp(float64(v - max)))
		sum += exps[i]
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

func SoftmaxWithTemperature(logits []float32, temperature float32) []float32 {
	scaled := make([]float32, len(logits))
	for i, v := range logits {
		scaled[i] = v / temperature
	}
	return Softmax(scaled)
}

// TopKLogits keeps the k largest logits (ties at the threshold are kept), -inf for the rest.
func TopKLogits(logits []float32, k int) []float32 {
	sorted := append([]float32(nil), logits...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
	threshold := sorted[k-1]
	out := make([]float32, len(logits))
	for i, v := range logits {
		if v < threshold {
			out[i] = float32(math.Inf(-1))
		} else {
			out[i] = v
		}
	}
	return out
}

// SampleIndex draws one index in proportion to probas: inverse-CDF sampling. One uniform number u
// in [0, 1); walk the cumulative sum and stop at the first entry where it passes u. Tokens with
// probability 0 can never be picked.
func SampleIndex(probas []float32, rng *attention.Rng) int {
	u := rng.NextF64()
	cumulative, last := 0.0, 0
	for i, p := range probas {
		if p > 0 {
			last = i
		}
		cumulative += float64(p)
		if u < cumulative {
			return i
		}
	}
	return last // rounding left the total a hair below u
}

// Generate is like GenerateTextSimple but with temperature and top-k (topK 0 = off). temperature 0 means greedy.
func Generate(m *gpt.Model, ids []int, maxNewTokens, contextSize int, rng *attention.Rng, temperature float32, topK int) []int {
	ids = append([]int(nil), ids...)
	for n := 0; n < maxNewTokens; n++ {
		start := 0
		if len(ids) > contextSize {
			start = len(ids) - contextSize
		}
		logits := m.Forward(ids[start:])
		last := append([]float32(nil), logits.Row(logits.Rows-1)...) // scores for the next token
		if topK > 0 {
			last = TopKLogits(last, topK)
		}
		next := 0
		if temperature > 0 {
			next = SampleIndex(SoftmaxWithTemperature(last, temperature), rng)
		} else {
			for i, v := range last {
				if v > last[next] {
					next = i
				}
			}
		}
		ids = append(ids, next)
	}
	return ids
}
