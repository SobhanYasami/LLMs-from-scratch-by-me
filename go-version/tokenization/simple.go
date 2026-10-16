package tokenization

import "fmt"

// Step 3 - simple tokenizers: split text, then look each token up in the vocab.
//
//	encode: "Hello, world" -> ["Hello", ",", "world"] -> [1131, 5, 987]
//	decode: [1131, 5, 987] -> ["Hello", ",", "world"] -> "Hello, world"
//
// Tip: compare the two types. V1 fails on a word it never saw; V2 swaps it for
// <|unk|> and keeps going. That one difference is why real models use BPE.

// SimpleTokenizerV1 maps known tokens to IDs and fails on unknown tokens.
type SimpleTokenizerV1 struct {
	strToInt map[string]int // token -> ID, used by Encode
	intToStr map[int]string // ID -> token, used by Decode (the vocab, inverted)
}

func NewSimpleTokenizerV1(vocab map[string]int) *SimpleTokenizerV1 {
	intToStr := make(map[int]string, len(vocab))
	for s, i := range vocab {
		intToStr[i] = s
	}
	return &SimpleTokenizerV1{strToInt: vocab, intToStr: intToStr}
}

// encode is shared by V1 and V2. The only difference is what happens to a
// token that is not in the vocab: error (V1) or <|unk|> (V2).
func (t *SimpleTokenizerV1) encode(text string, unknownToUnk bool) ([]int, error) {
	var ids []int
	for _, token := range SplitText(text) {
		id, ok := t.strToInt[token]
		if !ok && unknownToUnk {
			id, ok = t.strToInt[Unknown]
		}
		if !ok {
			return nil, fmt.Errorf("token not in vocab: %q", token)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func (t *SimpleTokenizerV1) Encode(text string) ([]int, error) { return t.encode(text, false) }

func (t *SimpleTokenizerV1) Decode(ids []int) (string, error) {
	tokens := make([]string, 0, len(ids))
	for _, id := range ids {
		token, ok := t.intToStr[id]
		if !ok {
			return "", fmt.Errorf("id not in vocab: %d", id)
		}
		tokens = append(tokens, token)
	}
	return Detokenize(tokens), nil
}

// SimpleTokenizerV2 is V1, except unknown tokens become <|unk|>.
// It embeds V1, so Decode is inherited unchanged.
type SimpleTokenizerV2 struct {
	SimpleTokenizerV1
}

func NewSimpleTokenizerV2(vocab map[string]int) *SimpleTokenizerV2 {
	return &SimpleTokenizerV2{SimpleTokenizerV1: *NewSimpleTokenizerV1(vocab)}
}

func (t *SimpleTokenizerV2) Encode(text string) ([]int, error) { return t.encode(text, true) }
