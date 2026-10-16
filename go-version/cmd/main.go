// Demo runner for the educational tokenizers and data sampling.
//
// Run from go-version/:
//
//	go run ./cmd
//
// Pipeline mirrors python-version/main.py:
//
//	raw text -> SplitText -> BuildVocab -> SimpleTokenizerV1/V2 -> tiktoken (GPT-2 BPE)
//	-> sliding-window dataset
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/pkoukk/tiktoken-go"
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"

	"llm-from-scratch/tokenization"
)

// Run from go-version/, like python-version/main.py resolves paths from its own dir.
const dataPath = "../assets/the-verdict.txt"

const v1Text = `"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.`

var v2Text = "Hello, do you like tea? " + tokenization.EndOfText + " In the sunlit terraces of the palace."

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func formatIDs(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = fmt.Sprint(id)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func loadText() string {
	raw, err := os.ReadFile(dataPath)
	check(err)
	return string(raw)
}

func loadVocab() map[string]int {
	return tokenization.BuildVocab(tokenization.SplitText(loadText()))
}

// showVocabTail prints the last count vocab entries (the highest IDs).
func showVocabTail(vocab map[string]int, count int) {
	type entry struct {
		token string
		id    int
	}
	entries := make([]entry, 0, len(vocab))
	for t, id := range vocab {
		entries = append(entries, entry{t, id})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].id < entries[j].id })
	for _, e := range entries[len(entries)-count:] {
		fmt.Printf("('%s', %d)\n", e.token, e.id)
	}
}

// demoV1: V1 works only on words it has seen in the corpus.
func demoV1(vocab map[string]int) {
	t := tokenization.NewSimpleTokenizerV1(vocab)
	fmt.Println("version 1 tokenizer :")
	ids, err := t.Encode(v1Text)
	check(err)
	fmt.Println(formatIDs(ids))
	decoded, err := t.Decode(ids)
	check(err)
	fmt.Println(decoded)
	fmt.Println("-------------------------")
}

// demoV2: V2 maps unknown words to <|unk|> and handles <|endoftext|>.
func demoV2(vocab map[string]int) {
	t := tokenization.NewSimpleTokenizerV2(vocab)
	fmt.Println("version 2 tokenizer :")
	ids, err := t.Encode(v2Text)
	check(err)
	fmt.Println(formatIDs(ids))
	decoded, err := t.Decode(ids)
	check(err)
	fmt.Println(decoded)
	fmt.Println("---------------------")
}

func newBPE() *tiktoken.Tiktoken {
	// Offline loader: embeds the BPE files instead of downloading them.
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())
	enc, err := tiktoken.GetEncoding("r50k_base") // same vocab as GPT-2
	check(err)
	return enc
}

// demoBPE: byte pair encoding, GPT-2 vocabulary (~50k tokens). No <|unk|> needed.
func demoBPE() {
	enc := newBPE()
	fmt.Println("bpe tokenizer :")
	ids := enc.Encode(v2Text, []string{tokenization.EndOfText}, nil)
	fmt.Println(formatIDs(ids))
	fmt.Println(enc.Decode(ids))
	fmt.Println("---------------------")
}

// demoSampling: next-word prediction pairs, target is the input shifted by one.
func demoSampling() {
	enc := newBPE()
	text := loadText()
	// Drop the first 50 tokens: gives a slightly more interesting passage.
	sample := enc.Encode(text, nil, nil)[50:]

	const contextSize = 4
	fmt.Println("data sampling :")
	fmt.Println("x:", formatIDs(sample[:contextSize]))
	fmt.Println("y:     ", formatIDs(sample[1:contextSize+1]))
	for i := 1; i <= contextSize; i++ {
		fmt.Println(enc.Decode(sample[:i]), "---->", enc.Decode([]int{sample[i]}))
	}

	// The same idea, packaged: every window of the whole text, in batches.
	ds := tokenization.NewGPTDataset(enc.Encode(text, nil, nil), 4, 4)
	in, tg := ds.Get(0)
	fmt.Println("windows:", ds.Len(), "first pair:", formatIDs(in), formatIDs(tg))
	inputs, targets := ds.Batches(8)
	fmt.Println("first batch inputs:")
	for _, row := range inputs[0] {
		fmt.Println(" ", formatIDs(row))
	}
	fmt.Println("first batch targets:")
	for _, row := range targets[0] {
		fmt.Println(" ", formatIDs(row))
	}
	fmt.Println("---------------------")
}

func main() {
	vocab := loadVocab()
	showVocabTail(vocab, 5)
	demoV1(vocab)
	demoV2(vocab)
	demoBPE()
	demoSampling()
}
