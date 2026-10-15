package tokenizer

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

const (
	EndOfText = "<|endoftext|>"
	Unknown   = "<|unk|>"
)

// splitRe mirrors the Python pattern: captured delimiters are kept in the output.
var splitRe = regexp.MustCompile(`([,.?_!"()']|--|\s)`)

// punctSpaceRe removes the space that decode inserts before punctuation.
var punctSpaceRe = regexp.MustCompile(`\s+([,.?!"()'])`)

// splitKeep behaves like Python's re.split with a capturing group.
func splitKeep(re *regexp.Regexp, s string) []string {
	var out []string
	last := 0
	for _, m := range re.FindAllStringIndex(s, -1) {
		out = append(out, s[last:m[0]], s[m[0]:m[1]])
		last = m[1]
	}
	return append(out, s[last:])
}

// preprocess splits text into tokens and drops empty/whitespace-only pieces.
func preprocess(text string) []string {
	var out []string
	for _, item := range splitKeep(splitRe, text) {
		if t := strings.TrimSpace(item); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// BuildVocab maps each unique token to an integer ID (sorted order), then
// appends the special tokens.
func BuildVocab(text string) map[string]int {
	set := make(map[string]struct{})
	for _, t := range preprocess(text) {
		set[t] = struct{}{}
	}
	all := make([]string, 0, len(set)+2)
	for t := range set {
		all = append(all, t)
	}
	sort.Strings(all)
	all = append(all, EndOfText, Unknown)

	vocab := make(map[string]int, len(all))
	for i, t := range all {
		vocab[t] = i
	}
	return vocab
}

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
	tokens := preprocess(text)
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
	text := strings.Join(parts, " ")
	return punctSpaceRe.ReplaceAllString(text, "${1}"), nil
}

// SimpleTokenizerV2 replaces unknown tokens with <|unk|> before encoding.
type SimpleTokenizerV2 struct {
	SimpleTokenizerV1
}

func NewSimpleTokenizerV2(vocab map[string]int) *SimpleTokenizerV2 {
	return &SimpleTokenizerV2{SimpleTokenizerV1: *NewSimpleTokenizerV1(vocab)}
}

func (t *SimpleTokenizerV2) Encode(text string) ([]int, error) {
	tokens := preprocess(text)
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
