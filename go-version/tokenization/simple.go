package tokenization

import "fmt"

// Step 3 - simple tokenizers that combine splitting with the vocabulary.
//
// Both types expose Encode (text -> IDs) and Decode (IDs -> text).
// V1 fails on tokens outside its vocabulary; V2 replaces them with <|unk|>.

// SimpleTokenizerV1 maps known tokens to IDs and fails on unknown tokens.
type SimpleTokenizerV1 struct {
	strToInt map[string]int
	intToStr map[int]string
}

func NewSimpleTokenizerV1(vocab map[string]int) *SimpleTokenizerV1 {
	intToStr := make(map[int]string, len(vocab))
	for s, i := range vocab {
		intToStr[i] = s
	}
	return &SimpleTokenizerV1{strToInt: vocab, intToStr: intToStr}
}

func (t *SimpleTokenizerV1) Encode(text string) ([]int, error) {
	tokens := SplitText(text)
	ids := make([]int, 0, len(tokens))
	for _, s := range tokens {
		id, ok := t.strToInt[s]
		if !ok {
			return nil, fmt.Errorf("token not in vocab: %q", s)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func (t *SimpleTokenizerV1) Decode(ids []int) (string, error) {
	parts := make([]string, 0, len(ids))
	for _, i := range ids {
		s, ok := t.intToStr[i]
		if !ok {
			return "", fmt.Errorf("id not in vocab: %d", i)
		}
		parts = append(parts, s)
	}
	return Detokenize(parts), nil
}

// SimpleTokenizerV2 replaces unknown tokens with <|unk|> before encoding.
type SimpleTokenizerV2 struct {
	SimpleTokenizerV1
}

func NewSimpleTokenizerV2(vocab map[string]int) *SimpleTokenizerV2 {
	return &SimpleTokenizerV2{SimpleTokenizerV1: *NewSimpleTokenizerV1(vocab)}
}

func (t *SimpleTokenizerV2) Encode(text string) ([]int, error) {
	tokens := SplitText(text)
	ids := make([]int, 0, len(tokens))
	for _, s := range tokens {
		if _, ok := t.strToInt[s]; !ok {
			s = Unknown
		}
		id, ok := t.strToInt[s]
		if !ok {
			return nil, fmt.Errorf("token not in vocab: %q", s)
		}
		ids = append(ids, id)
	}
	return ids, nil
}
