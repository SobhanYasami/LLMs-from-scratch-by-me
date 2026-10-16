// Demo runner for the educational tokenizers and data sampling.
//
// Run from go-version/:
//
//	go run ./cmd         # all chapters
//	go run ./cmd ch03    # one chapter
//
// Tip: read main() first, then jump into tokenization/ following each demo.
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

// demoDataLoader: batches of windows. stride < maxLength overlaps windows.
func demoDataLoader() {
	enc := newBPE()
	ids := enc.Encode(loadText(), nil, nil)
	fmt.Println("data loader :")
	b := tokenization.NewDataLoader(tokenization.NewGPTDataset(ids, 4, 1), 1, false, true, 123).Epoch()
	fmt.Println(formatIDs(b[0].Inputs[0]), formatIDs(b[0].Targets[0])) // first window
	fmt.Println(formatIDs(b[1].Inputs[0]), formatIDs(b[1].Targets[0])) // second: shifted by one
	batches := tokenization.NewDataLoader(tokenization.NewGPTDataset(ids, 4, 4), 8, false, true, 123).Epoch()
	fmt.Printf("inputs shape: (%d, %d)\n", len(batches[0].Inputs), len(batches[0].Inputs[0]))
	fmt.Println("---------------------")
}

// demoEmbeddings: token IDs -> vectors, then add position vectors.
func demoEmbeddings() {
	enc := newBPE()
	const maxLength, outputDim = 4, 256
	ds := tokenization.NewGPTDataset(enc.Encode(loadText(), nil, nil), maxLength, maxLength)
	inputs := tokenization.NewDataLoader(ds, 8, false, true, 123).Epoch()[0].Inputs

	emb := tokenization.NewInputEmbedding(50257, maxLength, outputDim, 123)
	tokenEmb := make([][][]float32, len(inputs))
	for i, row := range inputs {
		tokenEmb[i] = emb.Token.Forward(row)
	}
	input := emb.Forward(inputs)
	fmt.Println("embeddings :")
	// Shapes are the thing to notice: [batch, tokens] grows a new axis of size D.
	fmt.Printf("token ids shape:      (%d, %d)\n", len(inputs), len(inputs[0]))
	fmt.Printf("token embeddings:     (%d, %d, %d)\n", len(tokenEmb), len(tokenEmb[0]), len(tokenEmb[0][0]))
	fmt.Printf("position embeddings:  (%d, %d)  <- one vector per position, shared by all rows\n", len(emb.Pos.Weight), len(emb.Pos.Weight[0]))
	fmt.Printf("input embeddings:     (%d, %d, %d)  <- token + position\n", len(input), len(input[0]), len(input[0][0]))
	fmt.Println("---------------------")
}

func runChapter2() {
	vocab := loadVocab()
	showVocabTail(vocab, 5)
	demoV1(vocab)
	demoV2(vocab)
	demoBPE()
	demoSampling()
	demoDataLoader()
	demoEmbeddings()
}

// main runs one chapter's demos (go run ./cmd ch02) or all when no argument is given.
func main() {
	chapters := os.Args[1:]
	if len(chapters) == 0 {
		chapters = []string{"ch02", "ch03"}
	}
	for _, c := range chapters {
		switch c {
		case "ch02":
			runChapter2()
		case "ch03":
			runChapter3()
		default:
			fmt.Fprintf(os.Stderr, "unknown chapter %q; use ch02 or ch03\n", c)
			os.Exit(1)
		}
	}
}
