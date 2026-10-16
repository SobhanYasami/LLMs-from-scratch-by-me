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


BOOK = "Build a Large Language Model (From Scratch), Sebastian Raschka"
RASBT = "https://github.com/rasbt/LLMs-from-scratch/tree/main"

PAGES = [
    # ------------------------------------------------------------------ start
    dict(slug="index", nav="Overview", title="Build an LLM from scratch", group="Start", body="""
<p class="lede">A tutorial and reference for the first two chapters of <i>Build a Large Language Model (From Scratch)</i>. Chapter 1 builds the mental model. Chapter 2 turns it into code, written three times: Python, Go and Rust, with the same layout and the same output.</p>
<div class="aim"><b>Aim.</b> Understand what a GPT-style model is, and how text becomes the numbers it trains on, by reading small programs instead of a framework.</div>
<h2>The map</h2>
<div class="flow">
<div><b>Chapter 1</b>What an LLM is, how it is trained, what a GPT looks like. No code.</div><span>&rarr;</span>
<div class="hot"><b>Chapter 2</b>Text &rarr; tokens &rarr; IDs &rarr; training pairs &rarr; vectors. All the code on this site.</div><span>&rarr;</span>
<div><b>Chapter 3+</b>Attention, the GPT model, training. Not covered yet.</div>
</div>
<h2>How to read this site</h2>
<ul>
<li>Read in order. Page titles carry the book's section numbers (1.1, 2.2, ...) so you can follow along with the PDF.</li>
<li>Every code block has Python / Go / Rust tabs. Your choice is remembered.</li>
<li>Each page has an <b>Aim</b>. Chapter 1 pages end with <b>check yourself</b> questions. Chapter 2 pages end with exercises.</li>
<li>The <a href="reference.html">Reference</a> page is a lookup table of every function in all three languages.</li>
</ul>
<h2>Run the code</h2>
<table><tr><th>Language</th><th>Command</th></tr>
<tr><td>Python</td><td><code>python-version/.venv/bin/python python-version/main.py</code></td></tr>
<tr><td>Go</td><td><code>cd go-version &amp;&amp; go run ./cmd</code></td></tr>
<tr><td>Rust</td><td><code>cd rust-version &amp;&amp; cargo run</code></td></tr></table>
<p>All three print the same token IDs. Embedding values differ (each language draws its own random weights) but shapes match.</p>
<h2>Sources</h2>
<p>Text follows the book and Raschka's companion repo, <a href="https://github.com/rasbt/LLMs-from-scratch">rasbt/LLMs-from-scratch</a>, in my own words. The code here is a separate re-implementation for learning in three languages. Buy the book for the full story.</p>
"""),

    # -------------------------------------------------------------- chapter 1
    dict(slug="ch01-what-is-llm", nav="1.1-1.2 What is an LLM?", title="1.1-1.2: What is an LLM, and what is it for?", group="Chapter 1: Understanding LLMs", body="""
<div class="aim"><b>Aim.</b> Place LLMs in the bigger picture (AI, machine learning, deep learning) and see what they are used for.</div>
<h2>A neural network that predicts the next word</h2>
<p>A <b>large language model</b> is a deep neural network trained on huge amounts of text to understand and generate human-like text. "Large" means two things: many <b>parameters</b> (the adjustable weights, often billions) and a very large training set. All those weights are tuned for one simple job: <b>predict the next word</b>. It is surprising how capable that job makes the model.</p>
<h2>Where LLMs sit</h2>
<div class="nest"><div><i>ARTIFICIAL INTELLIGENCE</i><div><i>MACHINE LEARNING</i><div><i>DEEP LEARNING</i><div>LLMs</div></div></div></div></div>
<ul>
<li><b>AI</b>: machines doing tasks that need human-like intelligence.</li>
<li><b>Machine learning</b>: learn rules from data instead of writing them. A spam filter learns from labelled emails instead of a hand-written rule list.</li>
<li><b>Deep learning</b>: machine learning with neural networks of three or more layers. It removes manual <i>feature extraction</i>: no expert has to decide that "free", "win" and many exclamation marks signal spam.</li>
<li><b>LLMs</b>: deep learning applied to text. Because they generate text, they are also called <i>generative AI</i>.</li>
</ul>
<div class="tip">Even deep learning needs labels in the usual setup (spam / not spam). LLM pretraining is different: the next word itself is the label. That is 1.6.</div>
<h2>Applications (1.2)</h2>
<p>Anything that parses or produces text: translation, summaries, sentiment analysis, writing articles or code, chatbots and assistants, and answering questions over large specialised document sets such as law or medicine.</p>
<h2>Why build one yourself?</h2>
<p>Coding an LLM is the best way to learn how it works and where it breaks. It also prepares you to pretrain or finetune open models on your own data, and custom models can beat general ones on their own domain (finance and medical question answering are the book's examples).</p>
<h2>Check yourself</h2>
<details class="qa"><summary>What does "large" refer to?</summary>Parameter count and training-set size.</details>
<details class="qa"><summary>What single task is an LLM trained on?</summary>Predicting the next word.</details>
<details class="qa"><summary>What does deep learning remove compared with classic machine learning?</summary>Manual feature extraction by human experts.</details>
"""),

    dict(slug="ch01-stages", nav="1.3, 1.5 Training stages & data", title="1.3 and 1.5: Pretraining, finetuning and data", group="Chapter 1: Understanding LLMs", body="""
<div class="aim"><b>Aim.</b> Learn the two-stage recipe (pretrain, then finetune) and why training data scale matters.</div>
<h2>Two stages</h2>
<div class="flow">
<div><b>Raw text</b>unlabelled, billions of words</div><span>&rarr;</span>
<div class="hot"><b>Pretraining</b>next-word prediction</div><span>&rarr;</span>
<div><b>Base / foundation model</b>completes text</div><span>&rarr;</span>
<div class="hot"><b>Finetuning</b>small labelled dataset</div><span>&rarr;</span>
<div><b>Assistant or classifier</b></div>
</div>
<ul>
<li><b>Pretraining</b> gives broad language knowledge. The result is a <i>base</i> or <i>foundation</i> model (GPT-3 is the classic example). It completes half-written sentences and can learn a new task from a few examples.</li>
<li><b>Finetuning</b> adapts it with labelled data. Two common kinds: <b>instruction finetuning</b> (instruction and answer pairs) and <b>classification finetuning</b> (texts and class labels, like spam / not spam).</li>
</ul>
<h2>Why so much data? (1.5)</h2>
<p>GPT-3 was pretrained on a mix of web crawl, books and Wikipedia, roughly 300 billion tokens sampled from the pool below. A <b>token</b> is the unit of text a model reads, roughly a word or punctuation mark. Chapter 2 is all about producing tokens.</p>
<table><tr><th>Dataset</th><th>What</th><th>Tokens</th><th>Share of training</th></tr>
<tr><td>CommonCrawl (filtered)</td><td>web crawl</td><td>410 B</td><td>60%</td></tr>
<tr><td>WebText2</td><td>web crawl</td><td>19 B</td><td>22%</td></tr>
<tr><td>Books1</td><td>book corpus</td><td>12 B</td><td>8%</td></tr>
<tr><td>Books2</td><td>book corpus</td><td>55 B</td><td>8%</td></tr>
<tr><td>Wikipedia</td><td>high-quality text</td><td>3 B</td><td>3%</td></tr></table>
<p>Pretraining is very expensive (GPT-3's estimated cloud cost was about $4.6 million). So this book trains a small model on a small text for learning, and shows how to load openly published weights instead of paying for pretraining.</p>
<h2>Check yourself</h2>
<details class="qa"><summary>Why is pretraining data called "raw"?</summary>It carries no labels, just ordinary text.</details>
<details class="qa"><summary>Name the two popular kinds of finetuning.</summary>Instruction finetuning and classification finetuning.</details>
<details class="qa"><summary>Why can open pretrained weights save you money?</summary>You skip the expensive pretraining stage and only finetune.</details>
"""),

    dict(slug="ch01-transformer", nav="1.4 Transformers & tasks", title="1.4: The transformer, and what LLMs can do", group="Chapter 1: Understanding LLMs", body="""
<div class="aim"><b>Aim.</b> Meet the transformer (encoder and decoder), see how BERT and GPT each use half of it, and learn zero-shot versus few-shot use.</div>
<h2>The original transformer</h2>
<p>Introduced in the 2017 paper <i>Attention Is All You Need</i> for machine translation. It has two parts:</p>
<div class="flow">
<div><b>Input text</b>"This is an example"</div><span>&rarr;</span>
<div class="hot"><b>Encoder</b>turns text into vectors that capture context</div><span>&rarr;</span>
<div class="hot"><b>Decoder</b>generates the output one word at a time</div><span>&rarr;</span>
<div><b>Output</b>"Das ist ein Beispiel"</div>
</div>
<p>Both halves are layers connected by <b>self-attention</b>: a way for the model to weigh how important each token is relative to every other token. That gives it long-range context. Chapter 3 implements it. Chapter 2 prepares the inputs it eats.</p>
<h2>BERT versus GPT</h2>
<div class="cmp">
<div><h3>BERT: the encoder half</h3>Trained on <b>masked word prediction</b>: guess hidden words inside a sentence. Strong at classification (sentiment, topic, toxic content).</div>
<div><h3>GPT: the decoder half</h3>Trained to <b>generate</b>: predict what comes next. Strong at completion, translation, summaries, writing and code. This is what the book builds.</div>
</div>
<h2>Zero-shot and few-shot</h2>
<ul>
<li><b>Zero-shot</b>: do a task never seen, with no examples in the prompt.</li>
<li><b>Few-shot</b>: you show a handful of examples inside the prompt, and the model follows the pattern. No retraining.</li>
</ul>
<div class="tip">"Transformer" and "LLM" are used interchangeably but are not the same. Transformers also do vision, and some LLMs use recurrent or convolutional designs, mostly chasing efficiency.</div>
<h2>Check yourself</h2>
<details class="qa"><summary>Which half of the transformer does GPT keep?</summary>The decoder.</details>
<details class="qa"><summary>What is BERT trained to do?</summary>Predict masked (hidden) words.</details>
<details class="qa"><summary>Difference between zero-shot and few-shot?</summary>Few-shot puts examples in the prompt; zero-shot gives none.</details>
"""),

    dict(slug="ch01-gpt", nav="1.6-1.7 GPT & the roadmap", title="1.6-1.7: GPT architecture and the roadmap", group="Chapter 1: Understanding LLMs", body="""
<div class="aim"><b>Aim.</b> See why next-word prediction needs no labels, why GPT is just a decoder, and where this book is heading.</div>
<h2>Self-supervised learning</h2>
<p>Take any text. The input is the words so far; the <b>label is simply the next word</b>, which the text already contains. The data labels itself, so any pile of text can train the model. This is why chapter 2 builds exactly one thing: pairs of (input tokens, same tokens shifted by one).</p>
<pre class="out">text:    The cat sat on the mat
input:   The cat sat on
target:      cat sat on the     (shifted by one)</pre>
<h2>GPT is a decoder only</h2>
<ul>
<li>GPT = <b>G</b>enerative <b>P</b>retrained <b>T</b>ransformer, from OpenAI (2018). ChatGPT's first model was GPT-3 finetuned on instructions.</li>
<li>It uses only the <b>decoder</b>: no encoder, strictly left to right.</li>
<li>It is <b>autoregressive</b>: each new word is picked using the words before it, and then becomes part of the next input.</li>
<li>Scale differs hugely: the original transformer stacked 6 blocks; GPT-3 has 96 layers and 175 billion parameters. Modern models such as Llama still use the same core ideas.</li>
</ul>
<h2>Emergent behaviour</h2>
<p>GPT is only trained to predict the next word, yet it also translates, classifies and summarises. Nobody taught it those tasks explicitly. Abilities that appear without being trained for are called <b>emergent behaviour</b>, a result of scale and diverse data.</p>
<h2>The roadmap (1.7)</h2>
<div class="flow">
<div class="hot"><b>Stage 1</b>Data preparation, attention, the model (ch. 2-4)</div><span>&rarr;</span>
<div><b>Stage 2</b>Pretrain a small GPT, evaluate it, load open weights (ch. 5)</div><span>&rarr;</span>
<div><b>Stage 3</b>Finetune to classify or follow instructions (ch. 6-7)</div>
</div>
<h2>Chapter 1 in six lines</h2>
<ol>
<li>LLMs predict the next word with deep neural networks.</li>
<li>Pretrain on raw text, then finetune on labelled data.</li>
<li>The transformer's key idea is attention.</li>
<li>GPT-style models keep only the decoder.</li>
<li>Next-word prediction gives surprising emergent skills.</li>
<li>Finetuned custom models can beat general ones on their task.</li>
</ol>
<p><a href="ch02-overview.html">On to chapter 2: turning text into model input &rarr;</a></p>
<h2>Check yourself</h2>
<details class="qa"><summary>Why does GPT need no hand-made labels?</summary>The next word in the text is the label (self-supervised).</details>
<details class="qa"><summary>What does "autoregressive" mean?</summary>Earlier outputs are fed back as inputs for later predictions.</details>
<details class="qa"><summary>What is an emergent behaviour?</summary>A skill the model shows without being explicitly trained for it, such as translation.</details>
"""),

    # -------------------------------------------------------------- chapter 2
    dict(slug="ch02-overview", nav="2.1 Overview & embeddings", title="Chapter 2 overview, and 2.1: Why embeddings?", group="Chapter 2: Working with text data", body="""
<div class="aim"><b>Aim.</b> Know the five steps that turn raw text into model input, and why the last one (vectors) is needed at all.</div>
<h2>The pipeline you will build</h2>
<div class="flow">
<div><b>2.2</b>split into tokens</div><span>&rarr;</span>
<div><b>2.3-2.4</b>token &rarr; ID, special tokens</div><span>&rarr;</span>
<div><b>2.5</b>byte pair encoding</div><span>&rarr;</span>
<div><b>2.6</b>sliding-window training pairs</div><span>&rarr;</span>
<div class="hot"><b>2.7-2.8</b>embeddings + positions</div>
</div>
<h2>2.1 Understanding word embeddings</h2>
<p>Neural networks do math on continuous numbers. Words are categories, so they cannot be fed in directly. An <b>embedding</b> maps a discrete thing (a word, image, document) to a point in a continuous vector space.</p>
<ul>
<li>Different data needs different embedding models: a text model cannot embed audio.</li>
<li>Word2Vec was an early method: words appearing in similar contexts get similar vectors, so "bird" words cluster apart from "city" words.</li>
<li>Vectors can be 2-D for plotting, but real models use hundreds or thousands of dimensions. GPT-2 small uses 768; the biggest GPT-3 uses 12,288. More dimensions capture more nuance at more compute cost.</li>
<li>LLMs <b>learn their own embeddings</b> as the first layer, updated during training, so they fit the data and task. (Sentence or document embeddings exist too, used for retrieval; not covered.)</li>
</ul>
<h2>The data</h2>
<p>We use <i>The Verdict</i> by Edith Wharton (public domain), 20,479 characters, saved as <code>assets/the-verdict.txt</code>. Tiny on purpose, so everything runs in seconds on a laptop.</p>
<h2>Summary of the chapter</h2>
<ul>
<li>Text becomes tokens, tokens become IDs, IDs become vectors.</li>
<li>Special tokens handle unknown words and document boundaries.</li>
<li>BPE (used by GPT-2 and GPT-3) breaks unknown words into sub-words, so it needs no unknown token.</li>
<li>A sliding window creates (input, target) pairs.</li>
<li>Embedding layers are lookup tables; position vectors are added so order is not lost.</li>
</ul>
<h2>Going further in Raschka's repo</h2>
<table><tr><th>Folder</th><th>What it adds</th></tr>
<tr><td><a href="RASBT/ch02/01_main-chapter-code">01_main-chapter-code</a></td><td>Notebooks and exercise solutions for this chapter</td></tr>
<tr><td><a href="RASBT/ch02/02_bonus_bytepair-encoder">02_bonus_bytepair-encoder</a></td><td>Compare different BPE implementations</td></tr>
<tr><td><a href="RASBT/ch02/03_bonus_embedding-vs-matmul">03_bonus_embedding-vs-matmul</a></td><td>Embedding layer equals one-hot vector times a weight matrix</td></tr>
<tr><td><a href="RASBT/ch02/04_bonus_dataloader-intuition">04_bonus_dataloader-intuition</a></td><td>The data loader explained with plain numbers</td></tr>
<tr><td><a href="RASBT/ch02/05_bpe-from-scratch">05_bpe-from-scratch</a></td><td>Train a GPT-2 style BPE tokenizer yourself</td></tr>
<tr><td><a href="RASBT/ch02/06_bonus_simple-tokenizer-v3">06_bonus_simple-tokenizer-v3</a></td><td>A further simple-tokenizer variant</td></tr></table>
""".replace('href="RASBT', 'href="' + RASBT)),

    dict(slug="ch02-2-tokenizing", nav="2.2 Tokenizing text", title="2.2: Tokenizing text", group="Chapter 2: Working with text data",
         src=files(["tokenization/preprocess.py"], ["tokenization/preprocess.go"], ["src/tokenization/preprocess.rs"]), body="""
<div class="aim"><b>Aim.</b> A model cannot read text, only numbers. Before numbering anything, cut the text into pieces called <i>tokens</i>.</div>
<p>A token here is a word or a punctuation mark. A regular expression finds the delimiters (punctuation, <code>--</code>, whitespace), and we <em>keep</em> the delimiters so punctuation becomes its own token. Whitespace pieces are then dropped.</p>
<pre class="out">"Hello, world!"  -&gt;  ['Hello', ',', 'world', '!']</pre>
<h2>How the book gets there</h2>
<ol>
<li>Split on whitespace only: <code>re.split(r'(\\s)', text)</code> leaves punctuation glued to words (<code>'Hello,'</code>).</li>
<li>Also split on <code>,</code> and <code>.</code>: now punctuation is separate, but empty strings and spaces remain.</li>
<li>Filter with <code>item.strip()</code> to drop whitespace pieces.</li>
<li>Add <code>? " ( ) _ !</code> and <code>--</code>. Result on the whole story: <b>4,649 tokens</b>.</li>
</ol>
<div class="tip">In Python, <code>re.split</code> keeps delimiters only when the pattern is in parentheses (a capture group). Go and Rust have no such feature, so they walk the matches by hand. Same result, more visible mechanism.</div>
<div class="tip">We do not lowercase: capitalisation helps a model tell names from common nouns and learn sentence structure. We drop whitespace for brevity; for code (where indentation matters) you would keep it.</div>
<h2>Code</h2>
{{code}}
<h2>Detokenize is lossy</h2>
<p>Decoding joins tokens with spaces, then removes the space before punctuation. Original spacing is not stored, so <code>detokenize(split(text))</code> only round-trips for simple text.</p>
<h2>Exercises</h2>
<ol><li>Feed a sentence with <code>--</code>. One token or two?</li><li>Remove the parentheses from the Python pattern. What happens to punctuation?</li></ol>
"""),

    dict(slug="ch02-3-token-ids", nav="2.3 Token IDs", title="2.3: Converting tokens into token IDs", group="Chapter 2: Working with text data",
         src=files(["tokenization/vocab.py", "tokenization/simple.py"], ["tokenization/vocab.go", "tokenization/simple.go"],
                   ["src/tokenization/vocab.rs", "src/tokenization/simple.rs"]), body="""
<div class="aim"><b>Aim.</b> Give every unique token a number (the <i>vocabulary</i>), then build a tokenizer with <code>encode</code> (text to IDs) and <code>decode</code> (IDs to text).</div>
<h2>The vocabulary</h2>
<p>Tokenize the whole training text, remove duplicates, sort alphabetically, number from 0. The story has <b>1,159</b> unique tokens.</p>
<pre class="out">('!', 0)  ('"', 1)  ("'", 2)  ...  ('Has', 49)  ('He', 50)</pre>
<ul>
<li><b>Sorted</b> so the same corpus always gives the same IDs. Sets and maps have random order in all three languages.</li>
<li>For decoding we also need the <b>inverse</b> map, ID to token.</li>
</ul>
<h2>The tokenizer, V1</h2>
<div class="flow">
<div><b>encode</b>text &rarr; tokens &rarr; IDs</div><span>&middot;</span>
<div><b>decode</b>IDs &rarr; tokens &rarr; join &rarr; tidy punctuation</div>
</div>
<p>Try it on a sentence from the story:</p>
<pre class="out">[1, 58, 2, 872, 1013, 615, 541, 763, 5, 1155, 608, 5, 1, 69, 7, 39, 873, 1136, 773, 812, 7]
" It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.</pre>
<h2>When it breaks</h2>
<p>Encode <code>"Hello, do you like tea?"</code> and V1 fails (<code>KeyError: 'Hello'</code> in Python, an <code>error</code> in Go, an <code>Err</code> in Rust). "Hello" is not in the story, so it is not in the vocabulary. A closed vocabulary cannot handle new words. <a href="ch02-4-special-tokens.html">Next page</a> patches that.</p>
<h2>Code (vocabulary and tokenizer)</h2>
{{code}}
<h2>Sample output</h2>
<pre class="out">('younger', 1156)
('your', 1157)
('yourself', 1158)
('&lt;|endoftext|&gt;', 1159)
('&lt;|unk|&gt;', 1160)</pre>
<p>(The last two entries are the special tokens from 2.4.)</p>
<h2>Exercise</h2>
<p>Stop sorting in <code>build_vocab</code>. Do decoded outputs change? Why or why not?</p>
"""),

    dict(slug="ch02-4-special-tokens", nav="2.4 Special tokens", title="2.4: Adding special context tokens", group="Chapter 2: Working with text data",
         src=files(["tokenization/simple.py"], ["tokenization/simple.go"], ["src/tokenization/simple.rs"]), body="""
<div class="aim"><b>Aim.</b> Handle unknown words and document boundaries by adding two special tokens to the vocabulary.</div>
<table><tr><th>Token</th><th>Role</th><th>ID in our vocab</th></tr>
<tr><td><code>&lt;|unk|&gt;</code></td><td>stands for any word not in the vocabulary</td><td>1160</td></tr>
<tr><td><code>&lt;|endoftext|&gt;</code></td><td>separates unrelated texts when concatenated for training, so the model does not treat two books as one story</td><td>1159</td></tr></table>
<p>They are appended <b>last</b> so existing word IDs do not shift. The vocabulary grows from 1,159 to <b>1,161</b>.</p>
<h2>SimpleTokenizerV2</h2>
<p>Same as V1, but before lookup any token not in the vocabulary is replaced by <code>&lt;|unk|&gt;</code>, so encoding never fails.</p>
<pre class="out">text    = "Hello, do you like tea? &lt;|endoftext|&gt; In the sunlit terraces of the palace."
ids     = [1160, 5, 362, 1155, 642, 1000, 10, 1159, 57, 1013, 981, 1009, 738, 1013, 1160, 7]
decoded = "&lt;|unk|&gt;, do you like tea? &lt;|endoftext|&gt; In the sunlit terraces of the &lt;|unk|&gt;."</pre>
<p><i>Hello</i> and <i>palace</i> were never in the story, so both became <code>&lt;|unk|&gt;</code> (ID 1160) and their meaning is lost. The <code>&lt;|endoftext|&gt;</code> separator became 1159.</p>
<h2>Other special tokens you will meet</h2>
<ul>
<li><code>[BOS]</code> beginning of sequence.</li>
<li><code>[EOS]</code> end of sequence, like <code>&lt;|endoftext|&gt;</code>.</li>
<li><code>[PAD]</code> pads shorter texts in a batch to equal length.</li>
</ul>
<div class="tip">GPT uses only <code>&lt;|endoftext|&gt;</code>, for separating <em>and</em> padding (padded positions are masked out later, so which token pads does not matter). GPT also has no <code>&lt;|unk|&gt;</code>: byte pair encoding on the next page replaces it.</div>
<h2>Code (same file as 2.3, now look at V2)</h2>
{{code}}
<div class="tip">V2 reuses V1: subclassing in Python, struct embedding in Go, wrapping in Rust. In Go and Rust both types share one private <code>encode</code> that takes an "unknown to unk" flag.</div>
"""),

    dict(slug="ch02-5-bpe", nav="2.5 Byte pair encoding", title="2.5: Byte pair encoding", group="Chapter 2: Working with text data",
         src=files(["main.py"], ["cmd/main.go"], ["src/main.rs"]), body="""
<div class="aim"><b>Aim.</b> Replace the word-level tokenizer with the one GPT-2, GPT-3 and the original ChatGPT use: sub-word pieces, so no <code>&lt;|unk|&gt;</code> is ever needed.</div>
<p>We do not implement BPE training here. We call <b>tiktoken</b> (OpenAI, written in Rust) with the GPT-2 vocabulary of <b>50,257</b> tokens. <code>&lt;|endoftext|&gt;</code> gets the last ID, <b>50256</b>.</p>
<pre class="out">[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 262, 20562, 13]
Hello, do you like tea? &lt;|endoftext|&gt; In the sunlit terraces of the palace.</pre>
<h2>How BPE avoids unknown words</h2>
<p>Training starts with every single character as a token, then repeatedly <b>merges the most frequent neighbouring pairs</b> into new tokens (<code>d</code> + <code>e</code> becomes <code>de</code>, then maybe <code>def</code>, and so on) up to a size limit. At encode time, an unfamiliar word is split into the largest known pieces, down to single characters or bytes if needed. Any string can be encoded.</p>
<div class="tip">tiktoken refuses special-token text by default, to prevent injection. Python passes <code>allowed_special</code> to opt in. Go and Rust use <code>r50k_base</code>, the same vocabulary as GPT-2 under its tiktoken name.</div>
<h2>Code (the BPE demo is <code>demo_bpe</code> / <code>demoBPE</code> in each runner)</h2>
{{code}}
<h2>Exercises</h2>
<ol>
<li>Book exercise 2.1: encode the made-up text <code>"Akwirw ier"</code>, print each ID, decode each ID on its own, then decode the whole list. Does it round-trip?</li>
<li>Encode <code>Zorblax</code>. How many tokens?</li>
</ol>
"""),

    dict(slug="ch02-6-sampling", nav="2.6 Sliding window", title="2.6: Data sampling with a sliding window", group="Chapter 2: Working with text data",
         src=files(["tokenization/dataset.py"], ["tokenization/dataset.go", "tokenization/dataloader.go"],
                   ["src/tokenization/dataset.rs", "src/tokenization/dataloader.rs"]), body="""
<div class="aim"><b>Aim.</b> Turn one long list of token IDs into training examples for <i>next-token prediction</i>.</div>
<p>BPE-encode the whole story: <b>5,145</b> tokens. Each example is a window of IDs (the <b>input</b>) and the same window shifted by one (the <b>target</b>). Every position in a window is a free question: "given these tokens, what comes next?"</p>
<pre class="out">x: [290, 4920, 2241, 287]
y:      [4920, 2241, 287, 257]

[290] ----&gt; 4920
[290, 4920] ----&gt; 2241
[290, 4920, 2241] ----&gt; 287
[290, 4920, 2241, 287] ----&gt; 257

as text:
 and ----&gt;  established
 and established ----&gt;  himself
 and established himself ----&gt;  in
 and established himself in ----&gt;  a</pre>
<p>Left of the arrow is what the model sees, right is what it must predict. (The book first drops the first 50 tokens for a more interesting passage.)</p>
<h2>Dataset and DataLoader</h2>
<ul>
<li><b>Dataset</b> (<code>GPTDatasetV1</code>): holds every (input, target) window as tensors. Python extends torch's <code>Dataset</code>; Go and Rust use plain slices.</li>
<li><b>max_length</b>: window size (the context). Tiny here (4); real training uses 256 or more.</li>
<li><b>stride</b>: how far the window slides. <code>stride=1</code> gives heavily overlapping windows; <code>stride == max_length</code> gives none.</li>
<li><b>DataLoader</b>: groups windows into <b>batches</b>, optionally shuffled, optionally dropping a short last batch.</li>
</ul>
<pre class="out">stride=1, batch_size=1:
[tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]   first window
[tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]   second: shifted by one

stride=4, batch_size=8: inputs shape (8, 4), no overlap, no word skipped</pre>
<div class="tip">The book uses <code>stride = max_length</code> for real batches: the whole text is used but nothing overlaps, since more overlap risks overfitting. Batch size is a trade-off: small uses less memory but gives noisier updates.</div>
<h2>Code</h2>
{{code}}
<div class="tip">The loop stops at <code>len - max_length</code> because the target reaches one token past the input.</div>
<h2>Exercises</h2>
<ol>
<li>Book exercise 2.2: run with <code>max_length=2, stride=2</code> and <code>max_length=8, stride=2</code>. How do the windows change?</li>
<li>Set <code>stride=1</code>. How many windows now, versus <code>stride=4</code>?</li>
</ol>
"""),

    dict(slug="ch02-7-embeddings", nav="2.7-2.8 Embeddings & positions", title="2.7-2.8: Token embeddings and word positions", group="Chapter 2: Working with text data",
         src=files(["tokenization/embedding.py"], ["tokenization/embedding.go"],
                   ["src/tokenization/embedding.rs", "src/tokenization/rng.rs"]), body="""
<div class="aim"><b>Aim.</b> Turn integer IDs into vectors the model can learn from, and put word order into those vectors.</div>
<h2>2.7 Token embeddings</h2>
<p>An <b>embedding layer</b> is a weight matrix with one row per vocabulary entry and one column per dimension. Converting an ID is a <b>row lookup</b>.</p>
<pre class="out">vocab_size = 6, output_dim = 3
weight matrix (random at start):
 id 0  [ 0.3374, -0.1778, -0.1690]
 id 1  [ 0.9178,  1.5810,  1.3010]
 id 2  [ 1.2753, -0.2010, -0.1606]
 id 3  [-0.4015,  0.9666, -1.1481]   &lt;- embedding of token 3
 id 4  [-1.1589,  0.3255, -0.6315]
 id 5  [-2.8400, -0.7849, -1.4096]

input_ids [2, 3, 5, 1]  ->  rows 2, 3, 5, 1  (a 4 x 3 matrix)</pre>
<ul>
<li>Weights start <b>random</b> and are tuned by backpropagation during training (chapter 5). That is why the layer is trainable and why vectors become meaningful.</li>
<li>An embedding layer is just a faster version of one-hot encoding times a matrix multiply (see Raschka's bonus folder 03).</li>
<li>With the real BPE vocabulary: <b>50,257 rows</b>, and the book uses <b>256 dimensions</b> (GPT-3 uses 12,288).</li>
</ul>
<h2>2.8 Encoding word positions</h2>
<p>The same ID always maps to the same vector, wherever it sits. Self-attention is also blind to order. So we add position information.</p>
<ul>
<li><b>Absolute</b> position embeddings: one vector per position (first, second, ...). GPT uses these, learned during training.</li>
<li><b>Relative</b> ones encode the distance between tokens and generalise better to unseen sequence lengths.</li>
</ul>
<pre class="out">token ids          (8, 4)         batch of 8 windows, 4 tokens each
token embeddings   (8, 4, 256)    every ID became a 256-d vector
position embeddings (4, 256)      one vector per position 0..3
input embeddings   (8, 4, 256)    token + position  &lt;- what the model reads</pre>
<p>The position table is looked up with <code>0, 1, ..., context_length - 1</code> and added to every window in the batch. Text longer than the context length must be truncated.</p>
<h2>Code</h2>
{{code}}
<div class="tip">Python uses <code>torch.nn.Embedding</code>. Go and Rust use nested slices and a small seeded random generator (<code>rng.rs</code> in Rust), so there are no heavy dependencies. Values differ across languages; shapes do not.</div>
<h2>Where next?</h2>
<p>These input embeddings are exactly what chapter 3 (attention) consumes.</p>
<h2>Exercise</h2>
<p>Change <code>output_dim</code> from 256 to 8 and print the shapes again. What changes, and what does not?</p>
"""),

    dict(slug="reference", nav="Reference", title="Reference: one table, three languages", group="Reference", body="""
<p class="lede">Every public piece, side by side. Names follow each language's convention.</p>
<table>
<tr><th>Book</th><th>Python</th><th>Go</th><th>Rust</th></tr>
<tr><td>2.2 split</td><td><code>split_text</code></td><td><code>SplitText</code></td><td><code>split_text</code></td></tr>
<tr><td>2.3 join</td><td><code>detokenize</code></td><td><code>Detokenize</code></td><td><code>detokenize</code></td></tr>
<tr><td>2.3 vocab</td><td><code>build_vocab(tokens)</code></td><td><code>BuildVocab(tokens)</code></td><td><code>build_vocab(&amp;tokens)</code></td></tr>
<tr><td>2.3-2.4 tokenizers</td><td><code>SimpleTokenizerV1/V2</code></td><td><code>NewSimpleTokenizerV1/V2</code></td><td><code>SimpleTokenizerV1/V2::new</code></td></tr>
<tr><td>2.3 encode / decode</td><td>raises <code>KeyError</code></td><td>returns <code>error</code></td><td>returns <code>Result</code></td></tr>
<tr><td>2.4 special tokens</td><td><code>END_OF_TEXT</code>, <code>UNK_TOKEN</code></td><td><code>EndOfText</code>, <code>Unknown</code></td><td><code>END_OF_TEXT</code>, <code>UNK_TOKEN</code></td></tr>
<tr><td>2.5 BPE</td><td><code>tiktoken.get_encoding("gpt2")</code></td><td><code>tiktoken.GetEncoding("r50k_base")</code></td><td><code>tiktoken_rs::r50k_base()</code></td></tr>
<tr><td>2.6 windows</td><td><code>GPTDatasetV1</code></td><td><code>NewGPTDataset</code></td><td><code>GptDataset::new</code></td></tr>
<tr><td>2.6 batches</td><td><code>create_dataloader_v1</code></td><td><code>NewDataLoader(...).Epoch()</code></td><td><code>DataLoader::new(...).epoch()</code></td></tr>
<tr><td>2.7-2.8 embeddings</td><td><code>InputEmbedding</code> (torch)</td><td><code>NewInputEmbedding</code></td><td><code>InputEmbedding::new</code></td></tr>
</table>
<h2>Layout, identical in each language</h2>
<pre class="out">&lt;lang&gt;/
  README.md
  main                 demo runner
  tokenization/
    preprocess         2.2
    vocab              2.3
    simple             2.3-2.4
    dataset            2.6 (+ dataloader)
    embedding          2.7-2.8</pre>
<p>Source comments say "Step 1-5". Map to the book: Step 1 = 2.2, Step 2 = 2.3, Step 3 = 2.3-2.4, Step 4 = 2.6, Step 5 = 2.7-2.8.</p>
<h2>Key numbers for chapter 2</h2>
<table>
<tr><th>Thing</th><th>Value</th></tr>
<tr><td>Characters in <i>The Verdict</i></td><td>20,479</td></tr>
<tr><td>Word-level tokens / vocabulary</td><td>4,649 / 1,159 (+2 special = 1,161)</td></tr>
<tr><td>BPE tokens in the story</td><td>5,145</td></tr>
<tr><td>GPT-2 BPE vocabulary, <code>&lt;|endoftext|&gt;</code> ID</td><td>50,257, 50256</td></tr>
<tr><td>Demo batch</td><td>8 windows x 4 tokens, 256 dimensions</td></tr>
</table>
<h2>Adding the next chapter</h2>
<p>Add a sibling module in each language (for example <code>attention</code>), in the book's order, with the same names and a demo in each <code>main</code>. Then add pages to <code>PAGES</code> in <code>docs/build.py</code> and run <code>python3 docs/build.py</code>.</p>
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
