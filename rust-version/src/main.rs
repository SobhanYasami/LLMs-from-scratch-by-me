//! Demo runner for the educational tokenizers and data sampling.
//!
//! Run from anywhere: `cargo run --manifest-path rust-version/Cargo.toml`
//!
//! Tip: read `main()` first, then jump into `tokenization/` following each demo.
//!
//! Pipeline mirrors python-version/main.py:
//!   raw text -> split_text -> build_vocab -> SimpleTokenizerV1/V2 -> tiktoken (GPT-2 BPE)
//!   -> sliding-window dataset

use std::collections::HashMap;
use std::fs;
use std::path::Path;

mod ch03;
mod ch04;

use llm_from_scratch::tokenization::{
    DataLoader, END_OF_TEXT, GptDataset, InputEmbedding, SimpleTokenizerV1, SimpleTokenizerV2,
    build_vocab, split_text,
};
use tiktoken_rs::CoreBPE;

const V1_TEXT: &str =
    r#""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."#;

fn v2_text() -> String {
    format!("Hello, do you like tea? {END_OF_TEXT} In the sunlit terraces of the palace.")
}

/// Resolved from the crate dir at compile time, so the cwd does not matter.
fn load_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/the-verdict.txt");
    fs::read_to_string(path).expect("read the-verdict.txt")
}

fn load_vocab() -> HashMap<String, usize> {
    build_vocab(&split_text(&load_text()))
}

/// Print the last `count` vocabulary entries (the highest IDs).
fn show_vocab_tail(vocab: &HashMap<String, usize>, count: usize) {
    let mut entries: Vec<_> = vocab.iter().collect();
    entries.sort_by_key(|(_, id)| **id);
    for (token, id) in &entries[entries.len() - count..] {
        println!("('{token}', {id})");
    }
}

/// V1 works only on words it has seen in the corpus.
fn demo_v1(vocab: &HashMap<String, usize>) {
    let tokenizer = SimpleTokenizerV1::new(vocab.clone());
    println!("version 1 tokenizer :");
    let ids = tokenizer.encode(V1_TEXT).unwrap();
    println!("{ids:?}");
    println!("{}", tokenizer.decode(&ids).unwrap());
    println!("-------------------------");
}

/// V2 maps unknown words to <|unk|> and handles the <|endoftext|> separator.
fn demo_v2(vocab: &HashMap<String, usize>) {
    let tokenizer = SimpleTokenizerV2::new(vocab.clone());
    println!("version 2 tokenizer :");
    let ids = tokenizer.encode(&v2_text()).unwrap();
    println!("{ids:?}");
    println!("{}", tokenizer.decode(&ids).unwrap());
    println!("---------------------");
}

/// GPT-2 vocabulary (~50k tokens); tiktoken calls it r50k_base.
fn bpe() -> CoreBPE {
    tiktoken_rs::r50k_base().expect("load r50k_base")
}

/// Byte pair encoding: sub-word pieces, so no <|unk|> is ever needed.
fn demo_bpe() {
    let tokenizer = bpe();
    println!("bpe tokenizer :");
    let ids = tokenizer.encode_with_special_tokens(&v2_text());
    println!("{ids:?}");
    println!("{}", tokenizer.decode(&ids).unwrap());
    println!("---------------------");
}

/// Next-word prediction pairs: the target is the input shifted by one.
fn demo_sampling() {
    let tokenizer = bpe();
    let text = load_text();
    let all_ids = tokenizer.encode_ordinary(&text);
    // Drop the first 50 tokens: gives a slightly more interesting passage.
    let sample = &all_ids[50..];

    let context_size = 4;
    println!("data sampling :");
    println!("x: {:?}", &sample[..context_size]);
    println!("y:      {:?}", &sample[1..=context_size]);
    for i in 1..=context_size {
        let context = tokenizer.decode(&sample[..i]).unwrap();
        let desired = tokenizer.decode(&sample[i..=i]).unwrap();
        println!("{context} ----> {desired}");
    }

    // The same idea, packaged: every window of the whole text, in batches.
    let dataset = GptDataset::new(&all_ids, 4, 4);
    println!(
        "windows: {}, first pair: {:?}",
        dataset.len(),
        dataset.get(0)
    );
    let (inputs, targets) = dataset.batches(8).next().unwrap();
    println!("first batch inputs:");
    for row in inputs {
        println!("  {row:?}");
    }
    println!("first batch targets:");
    for row in targets {
        println!("  {row:?}");
    }
    println!("---------------------");
}

/// Batches of windows. stride < max_length overlaps windows.
fn demo_dataloader() {
    let ids = bpe().encode_ordinary(&load_text());
    println!("data loader :");
    let overlapping = GptDataset::new(&ids, 4, 1);
    let b = DataLoader::new(&overlapping, 1, false, true, 123).epoch();
    println!("{:?} {:?}", b[0].inputs[0], b[0].targets[0]); // first window
    println!("{:?} {:?}", b[1].inputs[0], b[1].targets[0]); // second: shifted by one
    let dataset = GptDataset::new(&ids, 4, 4);
    let batches = DataLoader::new(&dataset, 8, false, true, 123).epoch();
    println!(
        "inputs shape: ({}, {})",
        batches[0].inputs.len(),
        batches[0].inputs[0].len()
    );
    println!("---------------------");
}

/// Token IDs -> vectors, then add position vectors.
fn demo_embeddings() {
    let tokenizer = bpe();
    let (max_length, output_dim) = (4, 256);
    let ids = tokenizer.encode_ordinary(&load_text());
    let dataset = GptDataset::new(&ids, max_length, max_length);
    let inputs = DataLoader::new(&dataset, 8, false, true, 123)
        .epoch()
        .remove(0)
        .inputs;

    let emb = InputEmbedding::new(50257, max_length, output_dim, 123);
    let token_embeddings: Vec<_> = inputs.iter().map(|row| emb.token.forward(row)).collect();
    let input_embeddings = emb.forward(&inputs);
    println!("embeddings :");
    // Shapes are the thing to notice: [batch, tokens] grows a new axis of size D.
    println!(
        "token ids shape:      ({}, {})",
        inputs.len(),
        inputs[0].len()
    );
    println!(
        "token embeddings:     ({}, {}, {})",
        token_embeddings.len(),
        token_embeddings[0].len(),
        token_embeddings[0][0].len()
    );
    println!(
        "position embeddings:  ({}, {})  <- one vector per position, shared by all rows",
        emb.pos.weight.len(),
        emb.pos.weight[0].len()
    );
    println!(
        "input embeddings:     ({}, {}, {})",
        input_embeddings.len(),
        input_embeddings[0].len(),
        input_embeddings[0][0].len()
    );
    println!("---------------------");
}

fn run_chapter2() {
    let vocab = load_vocab();
    show_vocab_tail(&vocab, 5);
    demo_v1(&vocab);
    demo_v2(&vocab);
    demo_bpe();
    demo_sampling();
    demo_dataloader();
    demo_embeddings();
}

/// `cargo run -- ch03` runs one chapter's demos; with no argument, all of them.
fn main() {
    let mut chapters: Vec<String> = std::env::args().skip(1).collect();
    if chapters.is_empty() {
        chapters = vec!["ch02".into(), "ch03".into(), "ch04".into()];
    }
    for chapter in &chapters {
        match chapter.as_str() {
            "ch02" => run_chapter2(),
            "ch03" => ch03::run(),
            "ch04" => ch04::run(),
            "ch04-124m" => ch04::run_124m(),
            other => {
                eprintln!("unknown chapter {other:?}; use ch02, ch03, ch04 or ch04-124m");
                std::process::exit(1);
            }
        }
    }
}
