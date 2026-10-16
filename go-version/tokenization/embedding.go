package tokenization

import "math/rand"

// Step 5 - token and positional embeddings.
//
// A token ID is just an integer. An embedding layer is a lookup table with one
// vector per ID, so every token becomes a vector the model can learn from:
//
//	ids [B][T]  --token embedding-->  [B][T][D]
//	positions 0..T-1 --pos embedding-> [T][D]  (added to every batch row)
//	input embeddings = token + position  -> [B][T][D]
//
// Why add positions? Attention (later chapters) treats its input as a bag of
// vectors, so word order must be put into the vectors themselves.
//
// There is no tensor library here: tensors are nested slices, and weights are
// random (normal, mean 0, std 1) and untrained, like torch's nn.Embedding at init.

// Embedding is a lookup table of numRows vectors of size dim.
type Embedding struct {
	Weight [][]float32
}

func NewEmbedding(numRows, dim int, rng *rand.Rand) *Embedding {
	w := make([][]float32, numRows)
	for i := range w {
		w[i] = make([]float32, dim)
		for j := range w[i] {
			w[i][j] = float32(rng.NormFloat64())
		}
	}
	return &Embedding{Weight: w}
}

// Forward looks up one vector per ID.
func (e *Embedding) Forward(ids []int) [][]float32 {
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = e.Weight[id]
	}
	return out
}

// InputEmbedding is a token embedding plus an absolute positional embedding.
type InputEmbedding struct {
	Token *Embedding
	Pos   *Embedding
}

func NewInputEmbedding(vocabSize, contextLength, dim int, seed int64) *InputEmbedding {
	rng := rand.New(rand.NewSource(seed))
	return &InputEmbedding{
		Token: NewEmbedding(vocabSize, dim, rng),
		Pos:   NewEmbedding(contextLength, dim, rng),
	}
}

// Forward takes ids of shape [batch][seqLen] and returns [batch][seqLen][dim].
func (m *InputEmbedding) Forward(ids [][]int) [][][]float32 {
	out := make([][][]float32, len(ids))
	for b, row := range ids {
		tok := m.Token.Forward(row)
		sum := make([][]float32, len(row))
		for t := range row {
			sum[t] = make([]float32, len(tok[t]))
			for d := range sum[t] {
				sum[t][d] = tok[t][d] + m.Pos.Weight[t][d]
			}
		}
		out[b] = sum
	}
	return out
}
