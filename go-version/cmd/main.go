package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"llm-from-scratch/internal/tokenizer"
)

// Run from go-version/ (go run ./cmd), like python-version/main.py runs from its own dir.
const dataPath = "../assets/the-verdict.txt"

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

func main() {
	raw, err := os.ReadFile(dataPath)
	check(err)
	vocab := tokenizer.BuildVocab(string(raw))

	// Print the last 5 vocab entries in ID order.
	type entry struct {
		token string
		id    int
	}
	entries := make([]entry, 0, len(vocab))
	for t, id := range vocab {
		entries = append(entries, entry{t, id})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].id < entries[j].id })
	for _, e := range entries[len(entries)-5:] {
		fmt.Printf("('%s', %d)\n", e.token, e.id)
	}

	// Version 1 tokenizer
	v1 := tokenizer.NewSimpleTokenizerV1(vocab)
	text := `"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.`
	fmt.Println("version 1 tokenizer :")
	ids, err := v1.Encode(text)
	check(err)
	fmt.Println(formatIDs(ids))
	decoded, err := v1.Decode(ids)
	check(err)
	fmt.Println(decoded)
	fmt.Println("-------------------------")

	// Version 2 tokenizer
	v2 := tokenizer.NewSimpleTokenizerV2(vocab)
	text1 := "Hello, do you like tea?"
	text2 := "In the sunlit terraces of the palace."
	text = strings.Join([]string{text1, text2}, " "+tokenizer.EndOfText+" ")
	fmt.Println("version 2 tokenizer :")
	ids, err = v2.Encode(text)
	check(err)
	fmt.Println(formatIDs(ids))
	decoded, err = v2.Decode(ids)
	check(err)
	fmt.Println(decoded)
	fmt.Println("---------------------")
}
