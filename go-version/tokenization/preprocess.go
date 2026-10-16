// Package tokenization holds the educational tokenizer building blocks.
//
// Files, in the order you learn them:
//   - preprocess.go: split raw text into tokens and join them back.
//   - vocab.go:      map each unique token to an integer ID.
//   - simple.go:     SimpleTokenizerV1 and SimpleTokenizerV2.
//   - dataset.go:    sliding-window (input, target) pairs for next-token prediction.
//   - dataloader.go: batching and shuffling of those pairs.
//   - embedding.go:  token + positional embeddings.
//
// Step 1 - split raw text into tokens. A token is a word or a punctuation mark.
//
//	"Hello, world!" -> ["Hello", ",", "world", "!"]
//
// Tip: start reading here, then vocab.go -> simple.go -> dataset.go ->
// dataloader.go -> embedding.go. Each file uses the one before it.
package tokenization

import (
	"regexp"
	"strings"
)

// Special tokens. They are not produced by splitting text; they are added on purpose.
const (
	EndOfText = "<|endoftext|>"
	Unknown   = "<|unk|>"
)

// Tip: the regex lists the delimiters. Keeping them (instead of dropping them)
// is what turns punctuation into tokens of its own.
// splitRe mirrors the Python pattern: captured delimiters are kept in the output.
var splitRe = regexp.MustCompile(`([,.?_!"()']|--|\s)`)

// punctSpaceRe removes the space that Detokenize inserts before punctuation.
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

// SplitText returns the tokens of text, with whitespace-only pieces removed.
func SplitText(text string) []string {
	var out []string
	for _, item := range splitKeep(splitRe, text) {
		if t := strings.TrimSpace(item); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// Detokenize joins tokens back into text, undoing the spaces added before
// punctuation. Not a perfect inverse: the original spacing is lost.
func Detokenize(tokens []string) string {
	return punctSpaceRe.ReplaceAllString(strings.Join(tokens, " "), "${1}")
}
