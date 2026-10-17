package gpt

// Steps 15-16 (book 4.6 and 4.7) - the GPT model and text generation.
//
//	token IDs [T]
//	  -> token embedding + position embedding        [T, D]
//	  -> N x transformer block                       [T, D]
//	  -> final layer norm                            [T, D]
//	  -> output head (Linear D -> vocab, no bias)    [T, vocab]   = "logits"
//
// The logits of the LAST position are the model's scores for the next token.
// GenerateTextSimple picks the best one, appends it, and repeats.
//
// Weights start as random numbers from the repo's shared generator
// (Normal(0, 0.02), biases 0, layer norm scale 1 / shift 0), in a fixed order,
// so every language builds exactly the same untrained model. Untrained means
// the generated text is gibberish; training is chapter 5.

import "llm-from-scratch/attention"

type Config struct {
	VocabSize     int     // BPE vocabulary of GPT-2
	ContextLength int     // most tokens the model can look at
	EmbDim        int     // size of every token vector
	NHeads        int     // attention heads per block
	NLayers       int     // transformer blocks
	DropRate      float32 // dropout (training only)
	QKVBias       bool    // bias in the query/key/value projections
}

var Config124M = Config{50257, 1024, 768, 12, 12, 0.1, false}

// ConfigTiny is the same architecture, small enough to run in a blink in any language.
var ConfigTiny = Config{50257, 16, 32, 4, 2, 0.1, false}

// Book exercise 4.2: bigger GPT-2 models differ only in the configuration.
type Size struct {
	Name                    string
	EmbDim, NLayers, NHeads int
}

var Sizes = []Size{
	{"gpt2-small", 768, 12, 12},
	{"gpt2-medium", 1024, 24, 16},
	{"gpt2-large", 1280, 36, 20},
	{"gpt2-xl", 1600, 48, 25},
}

type Model struct {
	Cfg            Config
	TokEmb, PosEmb Matrix
	Blocks         []*TransformerBlock
	FinalNorm      *LayerNorm
	OutHead        Matrix // [D, vocab]
}

func NewModel(cfg Config) *Model {
	if cfg.QKVBias {
		panic("this implementation has no query/key/value bias (chapter 6 adds it)")
	}
	m := &Model{Cfg: cfg,
		TokEmb:    attention.NewMatrix(cfg.VocabSize, cfg.EmbDim),
		PosEmb:    attention.NewMatrix(cfg.ContextLength, cfg.EmbDim),
		FinalNorm: NewLayerNorm(cfg.EmbDim),
		OutHead:   attention.NewMatrix(cfg.EmbDim, cfg.VocabSize),
	}
	for i := 0; i < cfg.NLayers; i++ {
		m.Blocks = append(m.Blocks, NewTransformerBlock(cfg))
	}
	return m
}

// NewInitializedModel builds a model and fills it from the shared generator.
// The ORDER is part of the recipe: token emb, position emb, then per block
// wq, wk, wv, wo, ff1, ff2, then the output head.
func NewInitializedModel(cfg Config, seed uint64) *Model {
	m := NewModel(cfg)
	rng := attention.NewRng(seed)
	fill := func(w Matrix) {
		for i := range w.Data {
			w.Data[i] = rng.Normal() * 0.02
		}
	}
	fill(m.TokEmb)
	fill(m.PosEmb)
	for _, b := range m.Blocks {
		for _, w := range []Matrix{b.Att.WQuery, b.Att.WKey, b.Att.WValue, b.Att.OutWeight, b.FF.Expand.W, b.FF.Shrink.W} {
			fill(w)
		}
	}
	fill(m.OutHead)
	return m
}

// Forward takes token IDs [T] and returns logits [T, vocab].
func (m *Model) Forward(ids []int) Matrix {
	x := attention.NewMatrix(len(ids), m.Cfg.EmbDim)
	for t, id := range ids {
		tok, pos, row := m.TokEmb.Row(id), m.PosEmb.Row(t), x.Row(t)
		for j := range row {
			row[j] = tok[j] + pos[j]
		}
	}
	for _, b := range m.Blocks {
		x = b.Forward(x)
	}
	return attention.MatMul(m.FinalNorm.Forward(x), m.OutHead)
}

// CountParameters adds up every weight the model holds.
func (m *Model) CountParameters() int {
	n := len(m.TokEmb.Data) + len(m.PosEmb.Data) + len(m.OutHead.Data) + 2*len(m.FinalNorm.Scale)
	for _, b := range m.Blocks {
		a := b.Att
		n += len(a.WQuery.Data) + len(a.WKey.Data) + len(a.WValue.Data) + len(a.OutWeight.Data) + len(a.OutBias)
		n += len(b.FF.Expand.W.Data) + len(b.FF.Expand.B) + len(b.FF.Shrink.W.Data) + len(b.FF.Shrink.B)
		n += 2 * (len(b.Norm1.Scale) + len(b.Norm2.Scale))
	}
	return n
}

// ParameterCount does the same by arithmetic only (no model is built).
func ParameterCount(cfg Config) int {
	v, c, d, n := cfg.VocabSize, cfg.ContextLength, cfg.EmbDim, cfg.NLayers
	attention := 4*d*d + d // wq, wk, wv, wo and the output bias
	feedForward := d*4*d + 4*d + 4*d*d + d
	norms := 2 * 2 * d // two layer norms per block, scale + shift each
	return v*d + c*d + n*(attention+feedForward+norms) + 2*d + d*v
}

// GenerateTextSimple is greedy decoding: ids are the tokens so far; the result has new IDs appended.
func GenerateTextSimple(m *Model, ids []int, maxNewTokens, contextSize int) []int {
	ids = append([]int(nil), ids...)
	for n := 0; n < maxNewTokens; n++ {
		start := 0
		if len(ids) > contextSize { // keep only what fits in the context
			start = len(ids) - contextSize
		}
		logits := m.Forward(ids[start:])
		last := logits.Row(logits.Rows - 1) // scores for the NEXT token only
		// Softmax would turn these into probabilities, but it keeps the order,
		// so the biggest score is also the most likely token.
		best := 0
		for i, v := range last {
			if v > last[best] {
				best = i
			}
		}
		ids = append(ids, best)
	}
	return ids
}
