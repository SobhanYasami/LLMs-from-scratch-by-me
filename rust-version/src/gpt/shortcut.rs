//! Step 13 (book 4.4) - shortcut connections, and why they matter.
//!
//! A shortcut (residual) connection adds a layer's input to its output:
//!
//! ```text
//! x = x + layer(x)        instead of        x = layer(x)
//! ```
//!
//! During training the gradient of the loss must travel backwards through every
//! layer. Without shortcuts it shrinks a little at each layer (the vanishing
//! gradient problem), so the first layers barely learn. With shortcuts the
//! gradient also has a short path straight back, so it stays healthy.
//!
//! Python gets the gradients from torch's autograd. Here there is no autograd,
//! so this file does the backward pass by hand, in f64: the chain rule, layer by
//! layer, from the loss back to the first layer. It reproduces the book's numbers.

/// Linear + GELU. `w` is [out][in] like `nn.Linear`, flattened row by row.
pub struct ExampleLayer {
    pub out: usize,
    pub inp: usize,
    pub w: Vec<f64>,
    pub b: Vec<f64>,
}

pub struct ExampleDeepNeuralNetwork {
    pub layers: Vec<ExampleLayer>,
    pub use_shortcut: bool,
}

fn gelu64(v: f64) -> f64 {
    0.5 * v * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
}

/// The derivative of the tanh-approximated GELU.
fn gelu_grad(v: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let t = (c * (v + 0.044715 * v * v * v)).tanh();
    0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * v * v)
}

impl ExampleDeepNeuralNetwork {
    /// One forward and backward pass (loss = (output - target)^2). Returns the mean
    /// absolute gradient of each layer's weight matrix, first layer first.
    pub fn gradient_means(&self, x: &[f64], target: f64) -> Vec<f64> {
        let mut acts: Vec<Vec<f64>> = vec![x.to_vec()]; // input of every layer
        let mut pre: Vec<Vec<f64>> = vec![]; // Linear output (before GELU) of every layer
        let mut shortcut: Vec<bool> = vec![];
        for l in &self.layers {
            let input = acts.last().unwrap();
            let z: Vec<f64> = (0..l.out)
                .map(|i| {
                    l.b[i]
                        + (0..l.inp)
                            .map(|j| l.w[i * l.inp + j] * input[j])
                            .sum::<f64>()
                })
                .collect();
            let mut out: Vec<f64> = z.iter().map(|&v| gelu64(v)).collect();
            // A shortcut needs matching shapes, so the last (3 -> 1) layer has none.
            let sc = self.use_shortcut && out.len() == input.len();
            if sc {
                for (o, i) in out.iter_mut().zip(input) {
                    *o += i;
                }
            }
            pre.push(z);
            acts.push(out);
            shortcut.push(sc);
        }

        let mut grad = vec![2.0 * (acts.last().unwrap()[0] - target)]; // d loss / d output
        let mut means = vec![0.0; self.layers.len()];
        for li in (0..self.layers.len()).rev() {
            let (l, input) = (&self.layers[li], &acts[li]);
            // gradient at the Linear output
            let dz: Vec<f64> = (0..l.out)
                .map(|i| grad[i] * gelu_grad(pre[li][i]))
                .collect();
            let mut sum = 0.0;
            for dzi in &dz {
                for xj in input {
                    sum += (dzi * xj).abs(); // d loss / d W[i][j], every weight once
                }
            }
            means[li] = sum / (l.out * l.inp) as f64;
            // gradient handed to the layer below
            let mut next = vec![0.0; l.inp];
            for (j, nj) in next.iter_mut().enumerate() {
                for (i, dzi) in dz.iter().enumerate() {
                    *nj += l.w[i * l.inp + j] * dzi;
                }
                if shortcut[li] {
                    *nj += grad[j]; // the shortcut: gradient also flows straight through
                }
            }
            grad = next;
        }
        means
    }
}
