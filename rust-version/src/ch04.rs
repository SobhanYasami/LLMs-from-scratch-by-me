//! Chapter 4 demos: the GPT model, one building block at a time (book 4.1 to 4.7).
//!
//! Run from rust-version/:  `cargo run -- ch04`
//!                          `cargo run -- ch04-124m`   (the real 124M model)
//!
//! Tip: read the demos top to bottom. Numbers in 4.2, 4.3 and 4.4 match the book
//! (the 4.4 weights are the book's torch.manual_seed(123) ones, written out so
//! every language shares them). The model demos use random weights from the repo's
//! shared generator, so all five languages print the same numbers; they differ from
//! the book's torch output, and since the model is untrained its text is gibberish.

#![allow(clippy::excessive_precision)] // the book weights are written out in full

use llm_from_scratch::attention::Matrix;
use llm_from_scratch::gpt::{
    CONFIG_124M, CONFIG_TINY, Config, ExampleDeepNeuralNetwork, ExampleLayer, FeedForward,
    GptModel, LayerNorm, SIZES, gelu, generate_text_simple, parameter_count,
};
use llm_from_scratch::tokenization::rng::Rng;

/// Book 4.1: two short texts, already turned into token IDs by the GPT-2 tokenizer.
const BATCH: [[u32; 4]; 2] = [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]; // "Every effort moves you" / "Every day holds a"

/// Book 4.2: the outputs of a Linear(5, 6) + ReLU layer for two inputs.
fn layer_out() -> Matrix {
    Matrix::from_rows(&[
        &[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        &[0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000],
    ])
}

type Weights = (&'static [&'static [f64]], &'static [f64]);

/// Book 4.4: weights and biases of five Linear layers (3->3 four times, then 3->1), torch.manual_seed(123).
const SHORTCUT_LAYERS: [Weights; 5] = [
    (
        &[
            &[-0.23542964, 0.01912448, -0.28674594],
            &[0.21772662, -0.49193421, 0.42322308],
            &[-0.41964141, -0.45901766, -0.36482018],
        ],
        &[0.26147819, -0.21332639, 0.21605217],
    ),
    (
        &[
            &[-0.49001414, -0.35029206, -0.21198919],
            &[-0.11346072, -0.44043937, 0.37804362],
            &[-0.13615717, 0.18532233, 0.40826949],
        ],
        &[0.10756382, 0.15787685, 0.55729234],
    ),
    (
        &[
            &[-0.26039040, 0.18287641, -0.25687245],
            &[0.41260317, 0.46110451, -0.53230095],
            &[0.49285263, 0.27569306, 0.25159022],
        ],
        &[0.23768058, 0.47995073, -0.07623307],
    ),
    (
        &[
            &[-0.48826423, -0.16567004, -0.40661314],
            &[0.03816671, -0.10779562, -0.30968198],
            &[-0.05249349, 0.54698384, -0.04553875],
        ],
        &[0.01833092, -0.09004387, 0.09076351],
    ),
    (&[&[0.51442641, 0.35304558, 0.20493712]], &[0.12546431]),
];

/// 4 decimals; a tiny negative number is just zero.
fn fmt4(v: f32) -> String {
    let s = format!("{v:.4}");
    if s == "-0.0000" { "0.0000".into() } else { s }
}

fn row4(values: &[f32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|&v| fmt4(v))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn show4(label: &str, values: &[f32]) {
    println!("{label}: {}", row4(values));
}

/// 1234567 -> "1,234,567".
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

/// A float with two decimals and thousands separators.
fn commas_f(v: f64) -> String {
    let s = format!("{v:.2}");
    let (int, frac) = s.split_once('.').unwrap();
    format!("{}.{frac}", commas(int.parse().unwrap()))
}

fn build(cfg: Config) -> GptModel {
    GptModel::new_initialized(cfg, 123)
}

/// 4.1: token IDs in, one score per vocabulary word out, for every position.
fn demo_big_picture() {
    println!("== 4.1 the big picture (tiny GPT: emb 32, 4 heads, 2 blocks) ==");
    println!(
        "batch of token IDs, shape ({}, {}): [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]",
        BATCH.len(),
        BATCH[0].len()
    );
    let model = build(CONFIG_TINY);
    let logits = model.forward(&BATCH[0]); // the batch is processed one sequence at a time
    println!(
        "logits shape: ({}, {}, {})  <- [batch, tokens, vocab_size]",
        BATCH.len(),
        logits.rows,
        logits.cols
    );
    show4("logits[0][0], first 4 of 50257", &logits.row(0)[..4]);
}

/// 4.2: make each token's numbers mean 0, variance 1.
fn demo_layer_norm() {
    println!("== 4.2 layer normalization ==");
    let x = layer_out();
    let (mut means, mut vars) = ([0f32; 2], [0f32; 2]);
    for i in 0..2 {
        let row = x.row(i);
        means[i] = row.iter().sum::<f32>() / row.len() as f32;
        // the book first uses the unbiased variance
        vars[i] = row
            .iter()
            .map(|v| (v - means[i]) * (v - means[i]))
            .sum::<f32>()
            / (row.len() - 1) as f32;
    }
    let norm0: Vec<f32> = x
        .row(0)
        .iter()
        .map(|v| (v - means[0]) / (vars[0] as f64).sqrt() as f32)
        .collect();
    show4("mean per row", &means);
    show4("variance per row", &vars);
    show4("normalized, row 0", &norm0);
    let ln = LayerNorm::new(6).forward(&x);
    let (mut out_mean, mut out_var) = ([0f32; 2], [0f32; 2]);
    for i in 0..2 {
        out_mean[i] = ln.row(i).iter().sum::<f32>() / 6.0;
        out_var[i] = ln
            .row(i)
            .iter()
            .map(|v| (v - out_mean[i]) * (v - out_mean[i]))
            .sum::<f32>()
            / 6.0;
    }
    show4("LayerNorm output mean", &out_mean);
    show4("LayerNorm output variance (biased)", &out_var);
}

/// 4.3: a smooth ReLU, and the expand-then-shrink feed forward block.
fn demo_gelu_and_feed_forward() {
    println!("== 4.3 GELU and the feed forward network ==");
    let x: Vec<f32> = (0..7).map(|i| i as f32 - 3.0).collect();
    let g: Vec<f32> = x.iter().map(|&v| gelu(v)).collect();
    let r: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();
    show4("x   ", &x);
    show4("GELU", &g);
    show4("ReLU", &r);
    let mut rng = Rng::new(123);
    let mut ffn = FeedForward::new(32);
    ffn.expand.w = Matrix::randn(32, 128, 0.02, &mut rng);
    ffn.shrink.w = Matrix::randn(128, 32, 0.02, &mut rng);
    let out = ffn.forward(&Matrix::rand(3, 32, &mut rng));
    println!(
        "feed forward: input (2, 3, 32) -> output (2, {}, {}) (hidden layer is 4 x 32 = 128 wide)",
        out.rows, out.cols
    );
}

/// 4.4: shortcut connections keep the gradient alive.
fn demo_shortcut() {
    println!("== 4.4 shortcut connections (mean |gradient| of each layer's weights) ==");
    for use_shortcut in [false, true] {
        let layers = SHORTCUT_LAYERS
            .iter()
            .map(|(w, b)| ExampleLayer {
                out: w.len(),
                inp: w[0].len(),
                w: w.concat(),
                b: b.to_vec(),
            })
            .collect();
        let net = ExampleDeepNeuralNetwork {
            layers,
            use_shortcut,
        };
        let grads = net.gradient_means(&[1.0, 0.0, -1.0], 0.0);
        let parts: Vec<String> = grads.iter().map(|g| format!("{g:.6}")).collect();
        let label = if use_shortcut { "with   " } else { "without" };
        println!(
            "{label} shortcut: {}   (layer 0 ... layer 4)",
            parts.join("  ")
        );
    }
}

/// 4.5: attention + feed forward, each wrapped in norm and shortcut. Shape in = shape out.
fn demo_transformer_block() {
    println!("== 4.5 the transformer block ==");
    let model = build(CONFIG_TINY);
    let x = model.embed(&BATCH[0]);
    let out = model.blocks[0].forward(&x);
    println!(
        "input shape: ({}, {}, {})  output shape: ({}, {}, {})",
        BATCH.len(),
        x.rows,
        x.cols,
        BATCH.len(),
        out.rows,
        out.cols
    );
    show4("input[0][0], first 4 numbers", &x.row(0)[..4]);
    show4("output[0][0], first 4 numbers", &out.row(0)[..4]);
}

/// 4.6: how big is the model?
fn demo_parameters() {
    println!("== 4.6 model size ==");
    let tiny = build(CONFIG_TINY);
    println!(
        "tiny model: {} parameters (arithmetic says {})",
        commas(tiny.count_parameters()),
        commas(parameter_count(&CONFIG_TINY))
    );
    let cfg = CONFIG_124M;
    let total = parameter_count(&cfg);
    let head = cfg.vocab_size * cfg.emb_dim;
    let mb = |n: usize| n as f64 * 4.0 / (1024.0 * 1024.0);
    println!("GPT-2 small, no model built: {} parameters", commas(total));
    println!(
        "  minus the output head (weight tying in the original GPT-2): {}",
        commas(total - head)
    );
    println!("  float32 size: {:.2} MB", mb(total));
    let d = cfg.emb_dim;
    let ff = d * 4 * d + 4 * d + 4 * d * d + d;
    let att = 4 * d * d + d;
    println!(
        "exercise 4.1: feed forward {} vs attention {} parameters per block",
        commas(ff),
        commas(att)
    );
    println!("exercise 4.2: bigger models, same code, different configuration");
    println!(
        "  {:<20} {:>14} {:>14} {:>11}",
        "", "parameters", "weight-tied", "float32 MB"
    );
    for (name, emb_dim, n_layers, n_heads) in SIZES {
        let c = Config {
            emb_dim,
            n_layers,
            n_heads,
            ..cfg
        };
        let n = parameter_count(&c);
        println!(
            "  {:<20} {:>14} {:>14} {:>11}",
            name,
            commas(n),
            commas(n - c.vocab_size * c.emb_dim),
            commas_f(mb(n))
        );
    }
}

/// 4.7: predict one token, append it, repeat.
fn demo_generate(cfg: Config, label: &str) {
    println!("== 4.7 generating text ({label}) ==");
    let model = build(cfg);
    let tokenizer = tiktoken_rs::r50k_base().expect("load r50k_base");
    let encoded = tokenizer.encode_with_special_tokens("Hello, I am");
    println!("encoded: {encoded:?}");
    let out = generate_text_simple(&model, &encoded, 6, cfg.context_length);
    println!("output IDs: {out:?}");
    println!("output length: {}", out.len());
    println!("decoded: {}", tokenizer.decode(&out).unwrap());
    println!("(gibberish: the weights are random, nothing has been trained yet)");
}

pub fn run() {
    demo_big_picture();
    demo_layer_norm();
    demo_gelu_and_feed_forward();
    demo_shortcut();
    demo_transformer_block();
    demo_parameters();
    demo_generate(CONFIG_TINY, "tiny model");
}

/// The real 124M-parameter model: forward pass, then generation. Needs ~1 GB of memory.
pub fn run_124m() {
    let model = build(CONFIG_124M);
    println!("== GPT-2 small (124M) ==");
    println!("parameters built: {}", commas(model.count_parameters()));
    let logits = model.forward(&BATCH[0]);
    println!(
        "logits shape: ({}, {}, {})",
        BATCH.len(),
        logits.rows,
        logits.cols
    );
    show4("logits[0][0], first 4 of 50257", &logits.row(0)[..4]);
    demo_generate(CONFIG_124M, "124M model");
}
