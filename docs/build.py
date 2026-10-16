#!/usr/bin/env python3
"""Generate the GitHub Pages site into docs/.

    python3 docs/build.py

Prose lives in PAGES below. Code blocks are read from the real source files at
build time, so the site can never drift from the code. Commit the generated HTML.
"""

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
LANGS = [("python", "Python"), ("go", "Go"), ("rust", "Rust")]
HLJS = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0"
REPO = "https://github.com/SobhanYasami/LLMs-from-scratch-by-me"


def files(py, go, rs):
    return {"python": [f"python-version/{p}" for p in py],
            "go": [f"go-version/{p}" for p in go],
            "rust": [f"rust-version/{p}" for p in rs]}


PAGES = [
    dict(slug="index", nav="Overview", title="Build an LLM from scratch", group="Start", body="""
<p class="lede">A learning repo and living reference. One idea per step, written three times: Python, Go and Rust, with the same layout and the same output.</p>
<div class="aim"><b>Aim.</b> Understand how text becomes the numbers a GPT-style model trains on, by reading small programs instead of a framework. Follows <i>Build a Large Language Model (From Scratch)</i> by Sebastian Raschka, chapter 2.</div>
<h2>The pipeline</h2>
<pre class="out">raw text
  -&gt; split into tokens        (step 1)
  -&gt; token -&gt; ID             (step 2)
  -&gt; encode / decode          (step 3)
  -&gt; windows of (input, target)  (step 4)
  -&gt; vectors + positions      (step 5)   &lt;- what the model finally reads</pre>
<h2>How to read this site</h2>
<ul>
<li>Go through the steps in order. Each uses the one before it.</li>
<li>Every code block has Python / Go / Rust tabs. Your choice is remembered.</li>
<li>Each step has an <b>Aim</b>, the <b>code</b>, <b>sample output</b>, and <b>exercises</b>.</li>
<li>The <a href="reference.html">Reference</a> page is a lookup table of every function in all three languages.</li>
</ul>
<h2>Run it</h2>
<table><tr><th>Language</th><th>Command</th></tr>
<tr><td>Python</td><td><code>python-version/.venv/bin/python python-version/main.py</code></td></tr>
<tr><td>Go</td><td><code>cd go-version &amp;&amp; go run ./cmd</code></td></tr>
<tr><td>Rust</td><td><code>cd rust-version &amp;&amp; cargo run</code></td></tr></table>
<p>All three print the same token IDs. Embedding numbers differ (each language draws its own random weights) but shapes match.</p>
"""),
    dict(slug="01-split", nav="1. Split text", title="Step 1: Split text into tokens", group="Tutorial",
         src=files(["tokenization/preprocess.py"], ["tokenization/preprocess.go"], ["src/tokenization/preprocess.rs"]), body="""
<div class="aim"><b>Aim.</b> A model cannot read text, only numbers. Before numbering anything, cut the text into pieces called <i>tokens</i>.</div>
<p>A token here is a word or a punctuation mark. A regular expression finds the delimiters (punctuation, <code>--</code>, whitespace), and we <em>keep</em> the delimiters so punctuation becomes its own token. Whitespace pieces are then dropped.</p>
<pre class="out">"Hello, world!"  -&gt;  ['Hello', ',', 'world', '!']</pre>
<div class="tip">In Python, <code>re.split</code> keeps delimiters only when the pattern is in parentheses (a capture group). Go and Rust have no such feature, so they walk the matches by hand. Same result, more visible mechanism.</div>
<h2>Code</h2>
{{code}}
<h2>Detokenize is lossy</h2>
<p>Decoding joins tokens with spaces, then removes the space before punctuation. Original spacing is not stored, so <code>detokenize(split(text))</code> only round-trips for simple text.</p>
<h2>Exercises</h2>
<ol><li>Feed a sentence with <code>--</code>. One token or two?</li><li>Remove the parentheses from the Python pattern. What happens to punctuation?</li></ol>
"""),
    dict(slug="02-vocab", nav="2. Vocabulary", title="Step 2: Build a vocabulary", group="Tutorial",
         src=files(["tokenization/vocab.py"], ["tokenization/vocab.go"], ["src/tokenization/vocab.rs"]), body="""
<div class="aim"><b>Aim.</b> Give every unique token a number. The table token -&gt; ID is the bridge between text and the model.</div>
<pre class="out">{'!': 0, '"': 1, ..., '&lt;|endoftext|&gt;': 1159, '&lt;|unk|&gt;': 1160}</pre>
<ul>
<li><b>Sorted</b> so the same corpus always gives the same IDs. Sets and maps have random order in all three languages.</li>
<li><b>Special tokens go last</b> so word IDs do not move when you add one:
<code>&lt;|endoftext|&gt;</code> marks the boundary between documents, <code>&lt;|unk|&gt;</code> stands for any unseen word.</li>
</ul>
<h2>Code</h2>
{{code}}
<h2>Sample output</h2>
<pre class="out">('younger', 1156)
('your', 1157)
('yourself', 1158)
('&lt;|endoftext|&gt;', 1159)
('&lt;|unk|&gt;', 1160)</pre>
<h2>Exercise</h2>
<p>Stop sorting. Do decoded outputs change? Why or why not?</p>
"""),
    dict(slug="03-simple", nav="3. Encode & decode", title="Step 3: Simple tokenizers", group="Tutorial",
         src=files(["tokenization/simple.py"], ["tokenization/simple.go"], ["src/tokenization/simple.rs"]), body="""
<div class="aim"><b>Aim.</b> Combine split + vocabulary into <code>encode</code> (text to IDs) and <code>decode</code> (IDs to text), and see why a closed vocabulary breaks.</div>
<table><tr><th>Type</th><th>Unseen word</th></tr>
<tr><td>V1</td><td>error (<code>KeyError</code> in Python, <code>error</code> in Go, <code>Err</code> in Rust)</td></tr>
<tr><td>V2</td><td>replaced by <code>&lt;|unk|&gt;</code>, information is lost</td></tr></table>
<h2>Code</h2>
{{code}}
<h2>Sample output</h2>
<pre class="out">version 2 tokenizer :
[1160, 5, 362, 1155, 642, 1000, 10, 1159, 57, 1013, 981, 1009, 738, 1013, 1160, 7]
&lt;|unk|&gt;, do you like tea? &lt;|endoftext|&gt; In the sunlit terraces of the &lt;|unk|&gt;.</pre>
<p><i>Hello</i> and <i>palace</i> never appeared in the corpus, so they became <code>&lt;|unk|&gt;</code>. Byte pair encoding (next pages) removes this limit.</p>
<div class="tip">V2 reuses V1: subclassing in Python, struct embedding in Go, wrapping in Rust. In Go and Rust both types share one private <code>encode</code> that takes an "unknown to unk" flag.</div>
"""),
    dict(slug="bpe", nav="Bonus: BPE", title="Bonus: Byte pair encoding", group="Tutorial",
         src=files(["main.py"], ["cmd/main.go"], ["src/main.rs"]), body="""
<div class="aim"><b>Aim.</b> Replace the word-level tokenizer with the one real GPT models use: sub-word pieces, so no <code>&lt;|unk|&gt;</code> is ever needed.</div>
<p>We do not implement BPE training here. We call <b>tiktoken</b> (OpenAI) with the GPT-2 vocabulary, about 50,257 tokens. An unknown word is broken into smaller known pieces, down to single bytes if needed.</p>
<pre class="out">[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 262, 20562, 13]
Hello, do you like tea? &lt;|endoftext|&gt; In the sunlit terraces of the palace.</pre>
<div class="tip">tiktoken refuses special-token text by default, to prevent injection. Python passes <code>allowed_special</code> to opt in. Go and Rust use <code>r50k_base</code>, the same vocabulary as GPT-2 under its tiktoken name.</div>
<h2>Demo runners (BPE demo is <code>demo_bpe</code> / <code>demoBPE</code> / <code>demo_bpe</code>)</h2>
{{code}}
<h2>Exercise</h2>
<p>Encode a made-up word like <code>Zorblax</code>. How many tokens?</p>
"""),
    dict(slug="04-sampling", nav="4. Data sampling", title="Step 4: Sliding-window data sampling", group="Tutorial",
         src=files(["tokenization/dataset.py"], ["tokenization/dataset.go", "tokenization/dataloader.go"],
                   ["src/tokenization/dataset.rs", "src/tokenization/dataloader.rs"]), body="""
<div class="aim"><b>Aim.</b> Turn one long list of token IDs into training examples for <i>next-token prediction</i>.</div>
<p>Each example is a window of IDs (the <b>input</b>) and the same window shifted by one (the <b>target</b>). Every position in a window is a free question: "given these tokens, what comes next?"</p>
<pre class="out">ids    = [10, 11, 12, 13, 14, 15]
input  = [10, 11, 12, 13]
target = [11, 12, 13, 14]</pre>
<p><b>stride</b> is how far the window moves. <code>stride &lt; max_length</code>: windows overlap (more samples). <code>stride == max_length</code>: no overlap.</p>
<p>A <b>DataLoader</b> groups windows into batches. The GPU handles many at once; shuffling smooths training; dropping a short last batch keeps sizes equal.</p>
<h2>Code</h2>
{{code}}
<h2>Sample output</h2>
<pre class="out">x: [290, 4920, 2241, 287]
y:      [4920, 2241, 287, 257]
 and ----&gt;  established
 and established ----&gt;  himself
windows: 1286
inputs shape: (8, 4)</pre>
<div class="tip">The loop stops at <code>len - max_length</code> because the target reaches one token past the input.</div>
<h2>Exercise</h2>
<p>Set <code>stride=1</code>. How many windows now? Compare with <code>stride=4</code>.</p>
"""),
    dict(slug="05-embeddings", nav="5. Embeddings", title="Step 5: Token and positional embeddings", group="Tutorial",
         src=files(["tokenization/embedding.py"], ["tokenization/embedding.go"],
                   ["src/tokenization/embedding.rs", "src/tokenization/rng.rs"]), body="""
<div class="aim"><b>Aim.</b> Turn integer IDs into vectors the model can learn from, and put word order into those vectors.</div>
<pre class="out">ids [8, 4]        --token embedding-->  [8, 4, 256]
positions 0..3    --pos embedding--->   [4, 256]
input embeddings = token + position  ->  [8, 4, 256]</pre>
<ul>
<li>An <b>embedding layer</b> is a lookup table: one trainable vector per token ID.</li>
<li>Attention (later chapters) cannot see order by itself, so a second table gives one vector per <b>position</b>, added to the token vectors.</li>
<li>Weights start random and are learned in training. Here they stay untrained, so you can study shapes.</li>
</ul>
<h2>Code</h2>
{{code}}
<h2>Sample output</h2>
<pre class="out">token ids shape:      (8, 4)
token embeddings:     (8, 4, 256)
position embeddings:  (4, 256)  &lt;- one vector per position, shared by all rows
input embeddings:     (8, 4, 256)  &lt;- token + position</pre>
<div class="tip">Python uses <code>torch.nn.Embedding</code>. Go and Rust use nested slices and a small seeded random generator (<code>rng.rs</code> in Rust), so there are no heavy dependencies. Values differ across languages; shapes do not.</div>
<h2>Where next?</h2>
<p>These input embeddings are exactly what chapter 3 (attention) consumes.</p>
"""),
    dict(slug="reference", nav="Reference", title="Reference: one table, three languages", group="Reference", body="""
<p class="lede">Every public piece, side by side. Names follow each language's convention.</p>
<table>
<tr><th>Step</th><th>Python</th><th>Go</th><th>Rust</th></tr>
<tr><td>1 split</td><td><code>split_text</code></td><td><code>SplitText</code></td><td><code>split_text</code></td></tr>
<tr><td>1 join</td><td><code>detokenize</code></td><td><code>Detokenize</code></td><td><code>detokenize</code></td></tr>
<tr><td>2 vocab</td><td><code>build_vocab(tokens)</code></td><td><code>BuildVocab(tokens)</code></td><td><code>build_vocab(&amp;tokens)</code></td></tr>
<tr><td>3 tokenizers</td><td><code>SimpleTokenizerV1/V2</code></td><td><code>NewSimpleTokenizerV1/V2</code></td><td><code>SimpleTokenizerV1/V2::new</code></td></tr>
<tr><td>3 encode / decode</td><td>raises <code>KeyError</code></td><td>returns <code>error</code></td><td>returns <code>Result</code></td></tr>
<tr><td>4 windows</td><td><code>GPTDatasetV1</code></td><td><code>NewGPTDataset</code></td><td><code>GptDataset::new</code></td></tr>
<tr><td>4 batches</td><td><code>create_dataloader_v1</code></td><td><code>NewDataLoader(...).Epoch()</code></td><td><code>DataLoader::new(...).epoch()</code></td></tr>
<tr><td>5 embeddings</td><td><code>InputEmbedding</code> (torch)</td><td><code>NewInputEmbedding</code></td><td><code>InputEmbedding::new</code></td></tr>
<tr><td>special tokens</td><td><code>END_OF_TEXT</code>, <code>UNK_TOKEN</code></td><td><code>EndOfText</code>, <code>Unknown</code></td><td><code>END_OF_TEXT</code>, <code>UNK_TOKEN</code></td></tr>
</table>
<h2>Layout, identical in each language</h2>
<pre class="out">&lt;lang&gt;/
  README.md
  main                 demo runner
  tokenization/
    preprocess         step 1
    vocab              step 2
    simple             step 3
    dataset            step 4 (+ dataloader)
    embedding          step 5</pre>
<h2>Adding the next chapter</h2>
<p>Add a sibling module in each language (for example <code>attention</code>), in the book's order, with the same names and a demo in each <code>main</code>. Then add a page to <code>PAGES</code> in <code>docs/build.py</code> and run <code>python3 docs/build.py</code>.</p>
<h2>Language differences</h2>
<table>
<tr><th>Topic</th><th>Python</th><th>Go</th><th>Rust</th></tr>
<tr><td>Capturing split</td><td><code>re.split</code></td><td>manual</td><td>manual</td></tr>
<tr><td>V2 reuses V1</td><td>subclass</td><td>embed struct</td><td>wrap</td></tr>
<tr><td>Tensors</td><td>torch</td><td>nested slices</td><td>nested <code>Vec</code></td></tr>
<tr><td>GPT-2 BPE</td><td><code>tiktoken</code></td><td><code>tiktoken-go</code> + offline loader</td><td><code>tiktoken-rs</code></td></tr>
<tr><td>Random weights</td><td><code>torch</code></td><td><code>math/rand</code></td><td>built-in splitmix64</td></tr>
</table>
"""),
]

LANG_NAME = dict(LANGS)
HLJS_LANG = {"python": "python", "go": "go", "rust": "rust"}


def code_block(src):
    bar = "".join(f'<button data-lang="{k}">{n}</button>' for k, n in LANGS)
    panes = []
    for k, _ in LANGS:
        parts = []
        for rel in src[k]:
            text = (ROOT / rel).read_text(encoding="utf-8")
            parts.append(f'<div class="file">{html.escape(rel)}</div>'
                         f'<pre><code class="language-{HLJS_LANG[k]}">{html.escape(text)}</code></pre>')
        panes.append(f'<div class="pane" data-lang="{k}">{"".join(parts)}</div>')
    return f'<div class="tabs"><div class="tabbar">{bar}</div>{"".join(panes)}</div>'


def nav(current):
    out, group = [], None
    for p in PAGES:
        if p["group"] != group:
            group = p["group"]
            out.append(f"<h4>{group}</h4>")
        cls = ' class="on"' if p["slug"] == current else ""
        out.append(f'<a{cls} href="{p["slug"]}.html">{p["nav"]}</a>')
    return "\n".join(out)


def pager(i):
    prev = f'<a href="{PAGES[i-1]["slug"]}.html">&larr; {PAGES[i-1]["nav"]}</a>' if i > 0 else "<span></span>"
    nxt = f'<a href="{PAGES[i+1]["slug"]}.html">{PAGES[i+1]["nav"]} &rarr;</a>' if i < len(PAGES) - 1 else "<span></span>"
    return f'<div class="pager">{prev}{nxt}</div>'


def render(i, page):
    body = page["body"]
    if "{{code}}" in body:
        body = body.replace("{{code}}", code_block(page["src"]))
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(page["title"])}</title>
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="{HLJS}/styles/github.min.css" media="(prefers-color-scheme: light)">
<link rel="stylesheet" href="{HLJS}/styles/github-dark.min.css" media="(prefers-color-scheme: dark)">
</head><body><div class="layout">
<nav class="side"><a class="brand" href="index.html">LLM from scratch</a>
{nav(page["slug"])}
<h4>Source</h4><a href="{REPO}">GitHub repo</a></nav>
<main><h1>{html.escape(page["title"])}</h1>
{body}
{pager(i)}
</main></div>
<script src="{HLJS}/highlight.min.js"></script>
<script src="{HLJS}/languages/go.min.js"></script>
<script src="{HLJS}/languages/rust.min.js"></script>
<script src="site.js"></script>
</body></html>"""


for i, page in enumerate(PAGES):
    (OUT / f"{page['slug']}.html").write_text(render(i, page), encoding="utf-8")
print(f"wrote {len(PAGES)} pages to {OUT}")
