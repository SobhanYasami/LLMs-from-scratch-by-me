package tokenization

import "sort"

// Step 2 - build a vocabulary: a mapping from each token to a unique integer ID.
//
//	{"!": 0, "\"": 1, ..., "<|endoftext|>": 1159, "<|unk|>": 1160}
//
// Models only understand numbers, so this table is the bridge between text and IDs.
//
// Tip: a Go map has random iteration order. Sorting before numbering is what
// makes the IDs the same on every run.

// SpecialTokens are appended after the regular tokens so word IDs stay stable.
var SpecialTokens = []string{EndOfText, Unknown}

// BuildVocab assigns IDs to the unique tokens in sorted order, then appends the
// special tokens. Sorting makes IDs deterministic for a given corpus.
func BuildVocab(tokens []string) map[string]int {
	set := make(map[string]struct{})
	for _, t := range tokens {
		set[t] = struct{}{}
	}
	all := make([]string, 0, len(set)+len(SpecialTokens))
	for t := range set {
		all = append(all, t)
	}
	sort.Strings(all)
	all = append(all, SpecialTokens...)

	vocab := make(map[string]int, len(all))
	for i, t := range all {
		vocab[t] = i
	}
	return vocab
}
