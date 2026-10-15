# Python version: tokenizers from scratch

This is the Python implementation of the tokenizer chapter from *Build a Large Language Model (From Scratch)*. It starts from plain regular expressions and ends with a real byte pair encoding (BPE) tokenizer from `tiktoken`, so you can see what each layer adds.

The Go version in `../go-version` reproduces the same behavior. Compare the two if you want to see how the ideas translate across languages.

## Why tokenization?

A language model cannot read text. It reads sequences of integers. Tokenization is the pipeline that turns text into those integers:

```
"It's the last he painted."
        │  1. split          (regex)
        ▼
['It', "'", 's', 'the', 'last', 'he', 'painted', '.']
        │  2. vocabulary     (token -> ID)
        ▼
[ 58, 2, 872, 1013, 615, 541, 763, 5 ]
        │  model consumes IDs
        ▼
        ...
        │  decode: IDs -> tokens -> text
        ▼
"It' s the last he painted."
```

Every step has a tradeoff. The rest of this document walks through them in order.

## Layout

```
python-version/
├── main.py                  # demo runner: runs every tokenizer on sample text
├── README.md                # this file
└── tokenization/
    ├── __init__.py          # public API of the package
    ├── preprocess.py        # step 1: split text into tokens, and join back
    ├── vocab.py             # step 2: token -> integer ID mapping
    └── simple.py            # step 3: SimpleTokenizerV1 and V2
```

The data used throughout is `../assets/the-verdict.txt`, a short story by Edith Wharton.

## Running it

From the repository root:

```bash
python-version/.venv/bin/python python-version/main.py
```

or from inside `python-version/`:

```bash
.venv/bin/python main.py
```

You should see the five highest-ID vocabulary entries, the output of both simple tokenizers, and the BPE output. The script uses paths relative to its own file, so the current directory does not matter.

The `.venv` was created with Python 3.13. Install `tiktoken` into it if you recreate the environment:

```bash
python-version/.venv/bin/pip install tiktoken
```

Doctests in the package can be checked with:

```bash
python-version/.venv/bin/python -m doctest -v python-version/tokenization/preprocess.py
```

## Step 1: Split text (`tokenization/preprocess.py`)

`split_text` uses this regex:

```python
SPLIT_PATTERN = r'([,.?_!"()\']|--|\s)'
```

- The character class matches punctuation that should become its own token.
- `--` matches the double dash.
- `\s` matches whitespace so it can be removed.
- The **parentheses** around the pattern matter. `re.split` only keeps the separators when they are captured, so without them the punctuation would disappear.

After splitting, empty strings and whitespace-only pieces are dropped:

```python
>>> split_text('Hello, world!')
['Hello', ',', 'world', '!']
```

`detokenize` does the reverse. It joins tokens with spaces and removes the space before punctuation:

```python
>>> detokenize(['Hello', ',', 'world', '!'])
'Hello, world!'
```

This is lossy. The original spacing is not stored, so `detokenize(split_text(t))` only round-trips for simple text.

## Step 2: Build a vocabulary (`tokenization/vocab.py`)

The vocabulary maps every unique token to an integer:

```python
all_tokens = sorted(set(tokens))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: i for i, token in enumerate(all_tokens)}
```

Points to notice:

- **Sorting** makes IDs deterministic. The same corpus always produces the same vocabulary.
- **Special tokens** are appended at the end:
  - `<|endoftext|>` marks a boundary between independent documents.
  - `<|unk|>` stands for any token not in the vocabulary.

## Step 3: Encode and decode (`tokenization/simple.py`)

Both tokenizers share the same interface:

| Method | Input | Output |
|---|---|---|
| `encode(text)` | string | list of integer IDs |
| `decode(ids)` | list of integer IDs | string |

### `SimpleTokenizerV1`

Looks each token up in the vocabulary. Any word not in the corpus raises `KeyError`:

```python
tokenizer = SimpleTokenizerV1(vocab)
tokenizer.encode("Hello")   # KeyError: 'Hello' (it never appeared in the corpus)
```

This is the most direct version of the idea, and it shows the weakness of a closed vocabulary: the model cannot handle text it has never seen.

### `SimpleTokenizerV2`

Subclasses V1 and changes one thing. Unknown tokens become `<|unk|>` before lookup:

```python
tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode("Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.")
tokenizer.decode(ids)
# '<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.'
```

`Hello` and `palace` never appeared in the corpus, so they became `<|unk|>`. The information in those words is lost. This is the main limitation of word-level tokenizers, and BPE is the usual fix.

The decoded text keeps `<|unk|>` and `<|endoftext|>` as literal strings, so you can see where substitution happened.

## Step 4: Byte pair encoding (`main.py`, `demo_bpe`)

`tiktoken` is OpenAI's tokenizer library. The `gpt2` encoding is the same BPE vocabulary used by GPT-2, with about 50,000 tokens:

```python
tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
```

Compared with the simple tokenizers:

- **No `<|unk|>` needed.** BPE splits unknown words into smaller known pieces, so every string can be encoded.
- **Sub-word tokens.** Rare words become several tokens rather than one lost token.
- **`allowed_special`.** By default `tiktoken` refuses to encode special-token strings found in the input, to prevent accidental injection. Passing `allowed_special` opts in explicitly.

Run the demo to compare the IDs: `Hello` is `15496` in GPT-2's vocabulary, not in the corpus-based one.

## Concepts checklist

- [ ] Why a model needs integers, not text
- [ ] What a regex capture group changes in `re.split`
- [ ] Why the vocabulary is sorted
- [ ] What `<|endoftext|>` and `<|unk|>` are for
- [ ] Why V1 fails on unseen words and V2 does not
- [ ] What BPE solves that word-level tokenizers cannot

## Exercises

1. Add `"--"` (double dash) to a sentence and check whether it becomes one token or two.
2. Remove the parentheses from `SPLIT_PATTERN`. What happens to the punctuation?
3. Encode a sentence with only words from the corpus, then one with new words, using V1 and V2. Count the tokens in each result.
4. Change `build_vocab` so it does not sort. Does the decoded output change? Why or why not?
5. Use `tiktoken` to encode a made-up word such as `"Zorblax"`. How many tokens does it become?

## Notes for learners

- `main.py` is a script, not a library. The reusable code lives in `tokenization/`.
- Each step is a pure function or a small class, so you can test them one at a time.
- If your editor reports `tiktoken` as unresolved, check that the interpreter it uses is `python-version/.venv/bin/python`.
