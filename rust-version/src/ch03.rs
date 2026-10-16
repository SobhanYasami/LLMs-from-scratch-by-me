//! Chapter 3 demos: attention, one idea at a time (book 3.3 to 3.6).
//!
//! Run from rust-version/:  `cargo run -- ch03`
//!
//! Tip: read the demos top to bottom; each builds on the one before. The numbers
//! in 3.3 and 3.4 match the book exactly (3.4 uses the book's seed-123 weights,
//! hard-coded so every language shares them).

#![allow(clippy::excessive_precision)] // the book weights are written out in full

use llm_from_scratch::attention::{
    CausalAttention, Matrix, MultiHeadAttention, SelfAttention, simple_attention,
};
use llm_from_scratch::tokenization::rng::Rng;

/// "Your journey starts with one step." as 3-dimensional embeddings (book 3.3.1).
fn inputs() -> Matrix {
    Matrix::from_rows(&[
        &[0.43, 0.15, 0.89], // Your
        &[0.55, 0.87, 0.66], // journey
        &[0.57, 0.85, 0.64], // starts
        &[0.22, 0.58, 0.33], // with
        &[0.77, 0.25, 0.10], // one
        &[0.05, 0.80, 0.55], // step
    ])
}

/// The book's torch.manual_seed(123) weights (d_in=3, d_out=2), written out so no language needs torch.
fn book_weights() -> (Matrix, Matrix, Matrix) {
    (
        Matrix::from_rows(&[
            &[0.29611194, 0.51656228],
            &[0.25167072, 0.68855679],
            &[0.07397246, 0.86652195],
        ]),
        Matrix::from_rows(&[
            &[0.13657987, 0.10247904],
            &[0.18405646, 0.72644675],
            &[0.31525391, 0.68710667],
        ]),
        Matrix::from_rows(&[
            &[0.07563531, 0.19663817],
            &[0.31641197, 0.40174013],
            &[0.11856830, 0.82739538],
        ]),
    )
}

fn fmt_row(values: &[f32]) -> String {
    let parts: Vec<String> = values.iter().map(|v| format!("{v:.4}")).collect();
    format!("[{}]", parts.join(", "))
}

fn show_vec(label: &str, v: &[f32]) {
    println!("{label}: {}", fmt_row(v));
}

fn show_matrix(label: &str, m: &Matrix) {
    println!("{label}:");
    for i in 0..m.rows {
        println!("  {}", fmt_row(m.row(i)));
    }
}

/// 3.3: one query by hand, then all tokens at once.
fn demo_simple_attention() {
    let inputs = inputs();
    println!("== 3.3.1 simple attention for one query (x2, 'journey') ==");
    let query = Matrix::from_rows(&[inputs.row(1)]);
    let scores_2 = query.matmul(&inputs.transpose()); // dot product of every token with the query
    let weights_2 = scores_2.softmax_rows();
    let context_2 = weights_2.matmul(&inputs);
    show_vec("scores    ", scores_2.row(0));
    show_vec("weights   ", weights_2.row(0));
    show_vec("context z2", context_2.row(0));

    println!("== 3.3.2 all tokens at once ==");
    let (scores, weights, context) = simple_attention(&inputs);
    show_matrix("scores (T x T)", &scores);
    show_matrix("weights (rows sum to 1)", &weights);
    let sums: Vec<f32> = (0..weights.rows)
        .map(|i| weights.row(i).iter().sum())
        .collect();
    show_vec("row sums", &sums);
    show_matrix("context vectors (T x d)", &context);
}

/// 3.4: query/key/value with trainable (here: fixed) weights.
fn demo_self_attention() {
    let inputs = inputs();
    println!("== 3.4 self-attention with trainable weights (d_in=3, d_out=2) ==");
    let (wq, wk, wv) = book_weights();
    let x2 = Matrix::from_rows(&[inputs.row(1)]);
    let query_2 = x2.matmul(&wq);
    let keys = inputs.matmul(&wk);
    let values = inputs.matmul(&wv);
    show_vec("query_2", query_2.row(0));
    println!(
        "keys shape: ({}, {}), values shape: ({}, {})",
        keys.rows, keys.cols, values.rows, values.cols
    );
    let scores_2 = query_2.matmul(&keys.transpose());
    show_vec("scores_2 (q2 . keys)", scores_2.row(0));
    let weights_2 = scores_2
        .scale(1.0 / (keys.cols as f64).sqrt() as f32)
        .softmax_rows();
    show_vec("weights_2 (scaled)", weights_2.row(0));
    show_vec("context_2", weights_2.matmul(&values).row(0));
    let sa = SelfAttention {
        w_query: wq,
        w_key: wk,
        w_value: wv,
    };
    show_matrix("all context vectors (T x d_out)", &sa.forward(&inputs));
}

/// 3.5: mask the future, then dropout.
fn demo_causal_attention() {
    let inputs = inputs();
    println!("== 3.5 causal attention ==");
    let (wq, wk, wv) = book_weights();
    let ca = CausalAttention::new(wq, wk, wv, 0.0);
    show_matrix(
        "causal weights (upper triangle is 0)",
        &ca.attention_weights(&inputs, None),
    );
    show_matrix("causal context vectors", &ca.forward(&inputs, None));
    let mut ones = Matrix::new(6, 6);
    ones.data.fill(1.0);
    let dropped = ones.dropout(0.5, &mut Rng::new(123));
    show_matrix(
        "dropout 0.5 on ones(6, 6), seed 123 (kept values are doubled)",
        &dropped,
    );
}

/// 3.6: split the work across heads.
fn demo_multihead() {
    let inputs = inputs();
    println!("== 3.6 multi-head attention (d_in=3, d_out=2, heads=2) ==");
    let mha = MultiHeadAttention::random(3, 2, 2, &mut Rng::new(123));
    // A batch is just the same computation repeated per example; two copies = a batch of 2.
    let out = mha.forward(&inputs);
    println!(
        "input shape: (2, {}, {})  output shape: (2, {}, {})",
        inputs.rows, inputs.cols, out.rows, out.cols
    );
    show_matrix("output[0]", &out);
}

pub fn run() {
    demo_simple_attention();
    demo_self_attention();
    demo_causal_attention();
    demo_multihead();
}
