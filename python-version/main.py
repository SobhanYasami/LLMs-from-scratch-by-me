import re
from importlib.metadata import version
import tiktoken



# 1) Data preparation and sampling 
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab                              # A: token -> id
        self.int_to_str = {i: s for s, i in vocab.items()}   # B: id -> token

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):                                   # D
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)      # E: remove space before punctuation
        return text

import re


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"  # A: unknown words -> <|unk|>
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # B: remove space before punctuation
        return text
# =======================
# Load and tokenize the raw text
# =======================
with open("../assets/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# =======================
# Build the vocabulary (token -> integer ID)
# =======================

# Converting tokens into token IDs
# convert these tokens from a Python "string" to "an integer representation" to produce the so-called token IDs
# -----------------
# To map the previously generated tokens into token IDs, we have to build a
# so-called vocabulary first. This vocabulary defines how we map each unique
# word and special character to a unique integer

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
# print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# =======================
# Use the tokenizer
# =======================
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
print("version 1 tokenizer :")
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
print("-------------------------")


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print("version 2 tokenizer :")
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
print("---------------------")


# ===================================
# Byte pair encoding 
# ==================================
print("tiktoken version:", version("tiktoken"))


tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("byte pair tokenizer ==============")
print(integers)

strings = tokenizer.decode(integers)
print(strings)