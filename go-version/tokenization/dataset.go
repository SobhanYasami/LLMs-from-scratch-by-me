package tokenization

// Step 4 - sliding-window data sampling for next-token prediction.
//
//	ids      = [10 11 12 13 14 15]
//	window 0: input [10 11 12 13]  target [11 12 13 14]
//
// Why shifted by one? The model's job is: given the tokens so far, guess the
// next one. Every position in a window is a free training example.
//
// The target is the input shifted by one position. Stride is how far the window
// moves each step; stride == maxLength means no overlap.

// GPTDataset holds every (input, target) window of one tokenized text.
type GPTDataset struct {
	inputIDs  [][]int
	targetIDs [][]int
}

// NewGPTDataset cuts tokenIDs into windows, like Python's GPTDatasetV1.
func NewGPTDataset(tokenIDs []int, maxLength, stride int) *GPTDataset {
	d := &GPTDataset{}
	// Stop at len-maxLength: the target reaches one token past the input.
	for i := 0; i < len(tokenIDs)-maxLength; i += stride {
		d.inputIDs = append(d.inputIDs, tokenIDs[i:i+maxLength])
		d.targetIDs = append(d.targetIDs, tokenIDs[i+1:i+maxLength+1])
	}
	return d
}

func (d *GPTDataset) Len() int { return len(d.inputIDs) }

// Get returns the input and target of window idx.
func (d *GPTDataset) Get(idx int) (input, target []int) {
	return d.inputIDs[idx], d.targetIDs[idx]
}

// Batches groups consecutive windows, dropping a final short batch.
func (d *GPTDataset) Batches(batchSize int) (inputs, targets [][][]int) {
	for i := 0; i+batchSize <= d.Len(); i += batchSize {
		inputs = append(inputs, d.inputIDs[i:i+batchSize])
		targets = append(targets, d.targetIDs[i:i+batchSize])
	}
	return inputs, targets
}
