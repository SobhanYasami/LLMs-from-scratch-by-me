package tokenization

import "math/rand"

// Step 4b - batching. Mirrors torch's DataLoader.
//
// Why batch? The GPU processes many windows at once, which is much faster.
// Why shuffle? Windows in book order make training swing; random order smooths it.
// Why drop the last short batch? A different size can upset training.

// Batch is one group of windows: inputs[b] and targets[b] are one (input, target) pair.
type Batch struct {
	Inputs  [][]int
	Targets [][]int
}

// DataLoader serves a GPTDataset in batches.
type DataLoader struct {
	ds        *GPTDataset
	batchSize int
	shuffle   bool
	dropLast  bool
	rng       *rand.Rand
}

// NewDataLoader seeds the shuffle so runs are repeatable.
func NewDataLoader(ds *GPTDataset, batchSize int, shuffle, dropLast bool, seed int64) *DataLoader {
	return &DataLoader{ds, batchSize, shuffle, dropLast, rand.New(rand.NewSource(seed))}
}

// Epoch returns every batch once. With shuffle, each call uses a new order.
func (l *DataLoader) Epoch() []Batch {
	order := make([]int, l.ds.Len())
	for i := range order {
		order[i] = i
	}
	if l.shuffle {
		l.rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	}
	var batches []Batch
	for i := 0; i < len(order); i += l.batchSize {
		end := min(i+l.batchSize, len(order))
		if end-i < l.batchSize && l.dropLast {
			break
		}
		var b Batch
		for _, idx := range order[i:end] {
			in, tg := l.ds.Get(idx)
			b.Inputs = append(b.Inputs, in)
			b.Targets = append(b.Targets, tg)
		}
		batches = append(batches, b)
	}
	return batches
}
