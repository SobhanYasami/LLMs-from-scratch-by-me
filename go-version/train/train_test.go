package train

import (
	"math"
	"testing"

	"llm-from-scratch/attention"
	"llm-from-scratch/gpt"
	"llm-from-scratch/tokenization"
)

func tinyBatch() tokenization.Batch {
	return tokenization.Batch{Inputs: [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}}, Targets: [][]int{{2, 3, 4, 5}, {6, 7, 8, 9}}}
}

func TestUntrainedLossIsAboutLnVocab(t *testing.T) {
	m := gpt.NewInitializedModel(gpt.ConfigTiny, 123)
	got := CalcLossBatch(tinyBatch(), m)
	if want := float32(math.Log(float64(gpt.ConfigTiny.VocabSize))); math.Abs(float64(got-want)) > 0.1 {
		t.Fatalf("loss %v, expected about %v", got, want)
	}
}

func TestTrainingLowersTheLossOnAFixedBatch(t *testing.T) {
	m := gpt.NewInitializedModel(gpt.ConfigTiny, 123)
	grads, opt, b := m.ZerosLike(), NewAdamW(m, 3e-3, 0.1), tinyBatch()
	first := CalcLossBatch(b, m)
	for i := 0; i < 20; i++ {
		ZeroGrads(grads)
		LossAndGrads(m, grads, b)
		opt.StepWith(m, grads)
	}
	if last := CalcLossBatch(b, m); last >= first-1 {
		t.Fatalf("loss went from %v to %v: expected a clear drop", first, last)
	}
}

func TestGradientMatchesFiniteDifference(t *testing.T) {
	cfg := gpt.Config{VocabSize: 11, ContextLength: 4, EmbDim: 8, NHeads: 2, NLayers: 1}
	m := gpt.NewInitializedModel(cfg, 7)
	for _, tensor := range m.Tensors() { // bigger weights make the check sharper
		for i := range tensor {
			tensor[i] *= 20
		}
	}
	b := tokenization.Batch{Inputs: [][]int{{1, 2, 3, 4}}, Targets: [][]int{{2, 3, 4, 5}}}
	grads := m.ZerosLike()
	LossAndGrads(m, grads, b)
	probe := m.Blocks[0].FF.Expand.W.Data
	gradProbe := grads.Blocks[0].FF.Expand.W.Data
	for _, i := range []int{0, 5, 17, 40} {
		const eps = 1e-2
		old := probe[i]
		probe[i] = old + eps
		up := CalcLossBatch(b, m)
		probe[i] = old - eps
		down := CalcLossBatch(b, m)
		probe[i] = old
		numeric := (up - down) / (2 * eps)
		if math.Abs(float64(numeric-gradProbe[i])) > 5e-3+0.05*math.Abs(float64(numeric)) {
			t.Fatalf("weight %d: backward %v, finite difference %v", i, gradProbe[i], numeric)
		}
	}
}

func TestTopKAndTemperature(t *testing.T) {
	logits := []float32{4.51, 0.89, -1.9, 6.75, 1.63, 6.28}
	got := TopKLogits(logits, 3)
	if got[0] != 4.51 || !math.IsInf(float64(got[1]), -1) || got[3] != 6.75 {
		t.Fatalf("top-k kept the wrong logits: %v", got)
	}
	cold, hot := SoftmaxWithTemperature([]float32{1, 2, 3}, 0.1), SoftmaxWithTemperature([]float32{1, 2, 3}, 5)
	if cold[2] < 0.99 || hot[2] > 0.5 {
		t.Fatalf("temperature should sharpen below 1 and flatten above: %v %v", cold, hot)
	}
}

func TestSamplingNeverPicksAZeroProbability(t *testing.T) {
	rng := attention.NewRng(1)
	p := []float32{0, 0.5, 0, 0.5}
	for i := 0; i < 500; i++ {
		if idx := SampleIndex(p, rng); idx != 1 && idx != 3 {
			t.Fatalf("picked index %d, which has probability 0", idx)
		}
	}
}
