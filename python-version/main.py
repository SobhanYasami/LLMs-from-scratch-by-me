import re


# 1) Data preparation and sampling 

# =======================
# Tokenizing text
with open("../assets/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))
print(preprocessed[:30])

# =======================
# Converting tokens into token IDs
# convert these tokens from a Python "string" to "an integer representation" to produce the so-called token IDs
# -----------------
# To map the previously generated tokens into token IDs, we have to build a
# so-called vocabulary first. This vocabulary defines how we map each unique
# word and special character to a unique integer

all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50:
        break