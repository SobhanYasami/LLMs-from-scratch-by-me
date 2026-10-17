//! Chapter 5 demos: pretraining, one step at a time (book 5.1 to 5.5).
//!
//! Run from rust-version/:  `cargo run --release -- ch05`
//!                          `cargo run --release -- ch05-gpt2`   (OpenAI's weights, 5.5)
//!
//! Tip: read the demos top to bottom. The model is the tiny GPT of chapter 4 (emb 32, 2 blocks,
//! context 16) so that every language can TRAIN it in seconds; the book trains the 124M model on a
//! GPU. All five languages use the same weights, data order (no shuffling) and sampling generator,
//! so they print the same numbers. They differ from the book's torch output.

use std::path::Path;

use llm_from_scratch::gpt::{CONFIG_TINY, Config, GptModel, generate_text_simple};
use llm_from_scratch::tokenization::dataloader::{Batch, DataLoader};
use llm_from_scratch::tokenization::dataset::GptDataset;
use llm_from_scratch::tokenization::rng::Rng;
use llm_from_scratch::train::adamw::AdamW;
use llm_from_scratch::train::checkpoint::{load_checkpoint, save_checkpoint};
use llm_from_scratch::train::grad::{loss_and_grads, zero_grads};
use llm_from_scratch::train::loss::{calc_loss_loader, cross_entropy_sum, perplexity};
use llm_from_scratch::train::openai::{GPT2_SMALL_CONFIG, load_gpt2_weights};
use llm_from_scratch::train::sample::{
    generate, sample_index, softmax, softmax_with_temperature, top_k_logits,
};
use tiktoken_rs::CoreBPE;

const DATA_PATH: &str = "../assets/the-verdict.txt";
const GPT2_PATH: &str = "../assets/gpt2/model.safetensors";
const CFG: Config = Config {
    drop_rate: 0.0,
    ..CONFIG_TINY
}; // dropout stays off: our loop has no random masks
const BATCH: usize = 4;
const STRIDE: usize = 16;
const EPOCHS: usize = 2;
const EVAL_FREQ: usize = 24;
const EVAL_ITER: usize = 2;
const LR: f64 = 3e-3;
const WEIGHT_DECAY: f64 = 0.1;

fn fmt(v: f32, digits: usize) -> String {
    let s = format!("{v:.digits$}");
    if s.starts_with('-') && s[1..].chars().all(|c| c == '0' || c == '.') {
        s[1..].to_string()
    } else {
        s
    }
}

fn row(values: &[f32], digits: usize) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|&v| fmt(v, digits))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Text in single quotes with newlines flattened (same in every language).
fn quoted(text: &str) -> String {
    format!("'{}'", text.replace('\n', " "))
}

fn build(cfg: Config) -> GptModel {
    GptModel::new_initialized(cfg, 123)
}

fn bpe() -> CoreBPE {
    tiktoken_rs::r50k_base().expect("load r50k_base")
}

fn token_ids_to_text(bpe: &CoreBPE, ids: &[u32]) -> String {
    bpe.decode(ids).unwrap()
}

/// The 90% / 10% split of the story, cut into windows and batched in book order.
fn loaders(bpe: &CoreBPE, ctx: usize, batch: usize) -> (Vec<Batch>, Vec<Batch>) {
    let text = std::fs::read_to_string(DATA_PATH).expect("run from rust-version/");
    let split = (0.9 * text.chars().count() as f64) as usize;
    let make = |t: &str, drop_last: bool| {
        let ids = bpe.encode_ordinary(t);
        let ds = GptDataset::new(&ids, ctx, STRIDE);
        DataLoader::new(&ds, batch, false, drop_last, 123).epoch()
    };
    (make(&text[..split], true), make(&text[split..], false))
}

/// 5.1: from probabilities to a single number.
fn demo_loss(bpe: &CoreBPE) {
    println!("== 5.1 evaluating generative text models ==");
    let model = build(CFG);
    let inputs = [[16833u32, 3626, 6100], [40, 1107, 588]]; // ["every effort moves", "I really like"]
    let targets = [[3626u32, 6100, 345], [588, 428, 11311]]; // [" effort moves you", " really like chocolate"]
    println!("inputs : [[16833, 3626, 6100], [40, 1107, 588]]");
    println!("targets: [[3626, 6100, 345], [588, 428, 11311]]");
    println!(
        "probas shape: (2, 3, {})  <- [batch, tokens, vocab_size]",
        CFG.vocab_size
    );
    let mut log_probas = Vec::new();
    let mut ce_total = 0.0f32;
    for i in 0..2 {
        let logits = model.forward(&inputs[i]);
        ce_total += cross_entropy_sum(&logits, &targets[i]);
        let picked: Vec<f32> = (0..3)
            .map(|t| softmax(logits.row(t))[targets[i][t] as usize])
            .collect();
        println!(
            "probability of the target, text {}: {}",
            i + 1,
            row(&picked, 6)
        );
        log_probas.extend(picked.iter().map(|&p| (p as f64).ln() as f32));
    }
    println!("log probabilities: {}", row(&log_probas, 4));
    let avg = log_probas.iter().sum::<f32>() / log_probas.len() as f32;
    println!("average log probability: {}", fmt(avg, 4));
    println!("negative average (the loss): {}", fmt(-avg, 4));
    let loss = ce_total / 6.0;
    println!("cross_entropy of the whole batch at once: {}", fmt(loss, 4));
    println!(
        "perplexity: {}  (a model that guesses uniformly is torn between 50,257 tokens)",
        fmt(perplexity(loss), 0)
    );
    let (train, val) = loaders(bpe, CFG.context_length, BATCH);
    println!(
        "training batches: {} of {BATCH} x {} tokens, validation batches: {}",
        train.len(),
        CFG.context_length,
        val.len()
    );
    let (tl, vl) = (
        calc_loss_loader(&train, &model, None),
        calc_loss_loader(&val, &model, None),
    );
    println!(
        "untrained: training loss {}, validation loss {}  (a uniform guess gives ln(50257) = {})",
        fmt(tl, 3),
        fmt(vl, 3),
        fmt((50257f64).ln() as f32, 3)
    );
}

fn evaluate_model(model: &GptModel, train: &[Batch], val: &[Batch]) -> (f32, f32) {
    (
        calc_loss_loader(train, model, Some(EVAL_ITER)),
        calc_loss_loader(val, model, Some(EVAL_ITER)),
    )
}

fn generate_and_print_sample(model: &GptModel, bpe: &CoreBPE, start: &str) {
    let ids = bpe.encode_with_special_tokens(start);
    let out = generate_text_simple(model, &ids, 20, model.cfg.context_length);
    println!("{}", token_ids_to_text(bpe, &out).replace('\n', " "));
}

/// One training step: zero the old gradients, forward and backward, update the weights.
fn train_step(model: &mut GptModel, grads: &mut GptModel, opt: &mut AdamW, batch: &Batch) -> f32 {
    zero_grads(grads);
    let loss = loss_and_grads(model, grads, batch);
    opt.step(model, grads);
    loss
}

/// 5.2: train for a few epochs. Returns the trained model, its optimizer and the gradient buffer.
fn train_demo(bpe: &CoreBPE) -> (GptModel, AdamW, GptModel) {
    println!("== 5.2 training an LLM ==");
    let mut model = build(CFG);
    let mut grads = model.zeros_like();
    let mut opt = AdamW::new(&model, LR, WEIGHT_DECAY);
    let (train, val) = loaders(bpe, CFG.context_length, BATCH);
    let mut global_step = 0;
    for epoch in 0..EPOCHS {
        for batch in &train {
            train_step(&mut model, &mut grads, &mut opt, batch);
            if global_step % EVAL_FREQ == 0 {
                let (tl, vl) = evaluate_model(&model, &train, &val);
                println!(
                    "Ep {} (Step {:06}): Train loss {}, Val loss {}",
                    epoch + 1,
                    global_step,
                    fmt(tl, 3),
                    fmt(vl, 3)
                );
            }
            global_step += 1;
        }
        generate_and_print_sample(&model, bpe, "Every effort moves you");
    }
    (model, opt, grads)
}

const VOCAB: [&str; 9] = [
    "closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you",
];
#[allow(clippy::approx_constant)] // 6.28 is just the book's number
const NEXT_TOKEN_LOGITS: [f32; 9] = [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79];

/// 5.3: temperature and top-k.
fn demo_sampling(model: &GptModel, bpe: &CoreBPE) {
    println!("== 5.3 decoding strategies ==");
    let probas = softmax(&NEXT_TOKEN_LOGITS);
    println!("next-token probabilities: {}", row(&probas, 4));
    let greedy_pick = probas
        .iter()
        .enumerate()
        .fold(0, |b, (i, &p)| if p > probas[b] { i } else { b });
    println!("greedy pick: {}", VOCAB[greedy_pick]);
    for (label, t) in [("1", 1.0f32), ("0.1", 0.1), ("5", 5.0)] {
        let p = softmax_with_temperature(&NEXT_TOKEN_LOGITS, t);
        let mut rng = Rng::new(123);
        let mut counts = [0usize; 9];
        for _ in 0..1000 {
            counts[sample_index(&p, &mut rng)] += 1;
        }
        let shown: Vec<String> = counts
            .iter()
            .zip(VOCAB)
            .filter(|(c, _)| **c > 0)
            .map(|(c, w)| format!("{c} x {w}"))
            .collect();
        println!(
            "temperature {label}: P(pizza) = {}  1000 draws: {}",
            fmt(p[6], 4),
            shown.join(", ")
        );
    }
    let mut order: Vec<usize> = (0..9).collect();
    order.sort_by(|&a, &b| {
        NEXT_TOKEN_LOGITS[b]
            .partial_cmp(&NEXT_TOKEN_LOGITS[a])
            .unwrap()
    });
    let top: Vec<f32> = order[..3].iter().map(|&i| NEXT_TOKEN_LOGITS[i]).collect();
    println!(
        "top-3 logits: {} at positions {:?}",
        row(&top, 2),
        &order[..3]
    );
    let masked = top_k_logits(&NEXT_TOKEN_LOGITS, 3);
    let shown: Vec<String> = masked
        .iter()
        .map(|&v| {
            if v.is_infinite() {
                "'-inf'".into()
            } else {
                format!("'{}'", fmt(v, 2))
            }
        })
        .collect();
    println!("after masking with -inf: [{}]", shown.join(", "));
    println!("top-3 probabilities: {}", row(&softmax(&masked), 4));
    let start = bpe.encode_with_special_tokens("Every effort moves you");
    let ctx = model.cfg.context_length;
    let greedy = generate_text_simple(model, &start, 15, ctx);
    println!("greedy   : {}", quoted(&token_ids_to_text(bpe, &greedy)));
    let sampled = generate(model, &start, 15, ctx, &mut Rng::new(123), 1.4, Some(25));
    println!(
        "top-k 25, temperature 1.4: {}",
        quoted(&token_ids_to_text(bpe, &sampled))
    );
    let same = generate(model, &start, 15, ctx, &mut Rng::new(123), 0.0, None);
    println!("temperature 0 equals greedy: {}", same == greedy);
}

/// 5.4: save, load, and carry on exactly where we stopped.
fn demo_checkpoint(model: &mut GptModel, opt: &mut AdamW, grads: &mut GptModel, bpe: &CoreBPE) {
    println!("== 5.4 saving and loading ==");
    let path = Path::new("../assets/checkpoints/rust.bin");
    save_checkpoint(path, model, Some(opt)).unwrap();
    println!(
        "saved model and optimizer state: {} bytes",
        commas(std::fs::metadata(path).unwrap().len() as usize)
    );
    let (mut restored, restored_opt) = load_checkpoint(path, LR, WEIGHT_DECAY).unwrap();
    let mut restored_opt = restored_opt.unwrap();
    let x = [6109u32, 3626, 6100, 345];
    let (a, b) = (model.forward(&x), restored.forward(&x));
    let diff = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    println!(
        "largest logit difference between the two models: {}",
        fmt(diff, 6)
    );
    let (train, _) = loaders(bpe, CFG.context_length, BATCH);
    let mut restored_grads = restored.zeros_like();
    let continued: Vec<f32> = train[..3]
        .iter()
        .map(|b| train_step(model, grads, opt, b))
        .collect();
    let resumed: Vec<f32> = train[..3]
        .iter()
        .map(|b| train_step(&mut restored, &mut restored_grads, &mut restored_opt, b))
        .collect();
    println!("3 more steps, continued: losses {}", row(&continued, 3));
    println!("3 more steps, restored : losses {}", row(&resumed, 3));
    println!("same losses: {}", continued == resumed);
}

fn commas(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn run() {
    let bpe = bpe();
    demo_loss(&bpe);
    let (mut model, mut opt, mut grads) = train_demo(&bpe);
    demo_sampling(&model, &bpe);
    demo_checkpoint(&mut model, &mut opt, &mut grads, &bpe);
}

/// 5.5: OpenAI's GPT-2 small, loaded into our model. Needs assets/gpt2/model.safetensors.
pub fn run_gpt2() {
    let bpe = bpe();
    if !Path::new(GPT2_PATH).exists() {
        eprintln!("missing {GPT2_PATH}: download it first (command in src/train/openai.rs)");
        std::process::exit(1);
    }
    println!("== 5.5 loading pretrained weights from OpenAI ==");
    let model = load_gpt2_weights(Path::new(GPT2_PATH)).unwrap();
    println!(
        "loaded GPT-2 small: {} parameters (with the attention biases)",
        commas(model.count_parameters())
    );
    let start = bpe.encode_with_special_tokens("Every effort moves you");
    let ctx = GPT2_SMALL_CONFIG.context_length;
    let out = generate(&model, &start, 25, ctx, &mut Rng::new(123), 1.5, Some(50));
    println!("generated: {}", quoted(&token_ids_to_text(&bpe, &out)));
    let greedy = generate_text_simple(&model, &start, 25, ctx);
    println!("greedy   : {}", quoted(&token_ids_to_text(&bpe, &greedy)));
    let text = std::fs::read_to_string(DATA_PATH).expect("run from rust-version/");
    let split = (0.9 * text.chars().count() as f64) as usize;
    let ids = bpe.encode_ordinary(&text[split..]);
    let ds = GptDataset::new(&ids, 64, 64);
    let val = DataLoader::new(&ds, 2, false, true, 123).epoch();
    let loss = calc_loss_loader(&val, &model, Some(2));
    println!(
        "exercise 5.5: loss of the pretrained model on The Verdict (validation part, 2 batches of 2 x 64 tokens): {}",
        fmt(loss, 3)
    );
}
